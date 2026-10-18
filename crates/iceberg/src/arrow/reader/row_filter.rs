// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Predicate-driven row filtering for `ArrowReader`: constructing Arrow `RowFilter`s
//! from Iceberg predicates, row-group selection based on column statistics, and
//! row-selection via the Parquet page index. Also includes byte-range row-group
//! filtering used for file splitting.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_schema::{ArrowError, DataType, Field, Fields, Schema as ArrowSchema};
use parquet::arrow::arrow_reader::{ArrowPredicate, ArrowPredicateFn, RowSelection};
use parquet::arrow::{PARQUET_FIELD_ID_META_KEY, ProjectionMask};
use parquet::file::metadata::ParquetMetaData;
use parquet::schema::types::SchemaDescriptor;

use super::{ArrowReader, PredicateConverter};
use crate::arrow::create_primitive_array_repeated;
use crate::arrow::equality_delete_set::{EqDeleteSet, with_nulls};
use crate::arrow::record_batch_transformer::constants_map;
use crate::error::Result;
use crate::expr::BoundPredicate;
use crate::expr::visitors::bound_predicate_visitor::visit;
use crate::expr::visitors::page_index_evaluator::PageIndexEvaluator;
use crate::expr::visitors::row_group_metrics_evaluator::RowGroupMetricsEvaluator;
use crate::scan::FileScanTask;
use crate::spec::{Datum, Literal, NestedFieldRef, PrimitiveLiteral, Schema, StructType, Type};
use crate::{Error, ErrorKind};

impl ArrowReader {
    /// Converts the scan predicate to an Arrow predicate for the Parquet `RowFilter`.
    pub(super) fn get_arrow_predicate(
        predicates: &BoundPredicate,
        parquet_schema: &SchemaDescriptor,
        iceberg_field_ids: &HashSet<i32>,
        field_id_map: &HashMap<i32, usize>,
    ) -> Result<Box<dyn ArrowPredicate>> {
        // Collect Parquet column indices from field ids.
        // If the field id is not found in Parquet schema, it will be ignored due to schema evolution.
        let mut column_indices = iceberg_field_ids
            .iter()
            .filter_map(|field_id| field_id_map.get(field_id).cloned())
            .collect::<Vec<_>>();
        column_indices.sort();

        // The converter that converts `BoundPredicates` to `ArrowPredicates`
        let mut converter = PredicateConverter {
            parquet_schema,
            column_map: field_id_map,
            column_indices: &column_indices,
        };

        // After collecting required leaf column indices used in the predicate,
        // creates the projection mask for the Arrow predicates.
        let projection_mask = ProjectionMask::leaves(parquet_schema, column_indices.clone());
        let predicate_func = visit(&mut converter, predicates)?;
        Ok(Box::new(ArrowPredicateFn::new(
            projection_mask,
            predicate_func,
        )))
    }

    /// Builds one Arrow predicate per equality delete set, keeping the rows whose key is not
    /// in the set. Each predicate decodes only the set's key columns, and the `RowFilter`
    /// drops deleted rows before the remaining columns are decoded.
    ///
    /// A key column the data file does not store is resolved with the spec's column
    /// projection rules: the identity partition value, then the field's `initial-default`,
    /// then null. A key nested in a struct is still null in the rows where an enclosing
    /// struct is null. When no key column or enclosing struct is stored, the key is the same
    /// for every row, so the set either deletes the whole file or nothing.
    pub(super) fn get_equality_delete_predicates(
        delete_sets: &[Arc<EqDeleteSet>],
        task: &FileScanTask,
        parquet_schema: &SchemaDescriptor,
        file_schema: &ArrowSchema,
        field_id_map: &HashMap<i32, usize>,
    ) -> Result<Vec<Box<dyn ArrowPredicate>>> {
        let mut partition_constants: Option<HashMap<i32, Datum>> = None;
        let mut predicates: Vec<Box<dyn ArrowPredicate>> = Vec::with_capacity(delete_sets.len());

        for delete_set in delete_sets {
            let mut leaves = Vec::with_capacity(delete_set.fields().len());
            let mut sources = Vec::with_capacity(delete_set.fields().len());
            for field in delete_set.fields() {
                if let Some(&leaf) = field_id_map.get(&field.field_id) {
                    leaves.push(leaf);
                    sources.push(EqDeleteKeySource::Column(
                        parquet_schema.column(leaf).path().parts().to_vec(),
                    ));
                } else {
                    let constants = match &mut partition_constants {
                        Some(constants) => constants,
                        None => partition_constants.insert(Self::partition_constants(task)?),
                    };
                    let value = Self::missing_column_value(task, constants, field.field_id);
                    sources.push(Self::missing_key_source(
                        task,
                        field.field_id,
                        value,
                        file_schema,
                        &mut leaves,
                    ));
                }
            }

            if leaves.is_empty() {
                let columns = Self::eq_delete_key_columns(&sources, delete_set, None, 1)?;
                if delete_set.keep_mask(&columns)?.value(0) {
                    continue;
                }
            }
            leaves.sort_unstable();
            leaves.dedup();

            let delete_set = delete_set.clone();
            let predicate = move |batch: RecordBatch| {
                let columns = Self::eq_delete_key_columns(
                    &sources,
                    &delete_set,
                    Some(&batch),
                    batch.num_rows(),
                )
                .map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
                delete_set.keep_mask(&columns)
            };
            predicates.push(Box::new(ArrowPredicateFn::new(
                ProjectionMask::leaves(parquet_schema, leaves),
                predicate,
            )));
        }

        Ok(predicates)
    }

    fn partition_constants(task: &FileScanTask) -> Result<HashMap<i32, Datum>> {
        match (&task.partition_spec, &task.partition) {
            (Some(spec), Some(partition)) => constants_map(spec, partition, &task.schema),
            _ => Ok(HashMap::new()),
        }
    }

    /// The value every row reads for a column the data file does not store.
    fn missing_column_value(
        task: &FileScanTask,
        partition_constants: &HashMap<i32, Datum>,
        field_id: i32,
    ) -> Option<PrimitiveLiteral> {
        if let Some(datum) = partition_constants.get(&field_id) {
            return Some(datum.literal().clone());
        }
        match task
            .schema
            .field_by_id(field_id)
            .and_then(|field| field.initial_default.as_ref())
        {
            Some(Literal::Primitive(literal)) => Some(literal.clone()),
            _ => None,
        }
    }

    /// Resolves a key column the data file does not store. Its value is `value`, except in
    /// the rows where an enclosing struct is null: the deepest enclosing struct the file
    /// stores decides those rows, and one of its stored leaves is added to `leaves` so the
    /// struct is decoded. The file stores a struct if its physical schema holds a field with
    /// the struct's id, even when every child it stores was since dropped from the table
    /// schema. An enclosing struct the file does not store either reads as its
    /// `initial-default`, or as null when it has none, which makes the key null.
    fn missing_key_source(
        task: &FileScanTask,
        field_id: i32,
        value: Option<PrimitiveLiteral>,
        file_schema: &ArrowSchema,
        leaves: &mut Vec<usize>,
    ) -> EqDeleteKeySource {
        let ancestors = struct_ancestors(task.schema.as_struct(), field_id).unwrap_or_default();
        let stored = stored_struct(file_schema.fields(), &ancestors);
        let missing_from = stored.as_ref().map_or(0, |(depth, ..)| depth + 1);

        if value.is_none()
            || ancestors[missing_from..]
                .iter()
                .any(|ancestor| ancestor.initial_default.is_none())
        {
            return EqDeleteKeySource::Constant(None);
        }
        match stored {
            Some((_, struct_path, leaf)) => {
                leaves.push(leaf);
                EqDeleteKeySource::ConstantInStruct { struct_path, value }
            }
            None => EqDeleteKeySource::Constant(value),
        }
    }

    /// Materialises the key columns of `delete_set` for `num_rows` rows of `batch`.
    fn eq_delete_key_columns(
        sources: &[EqDeleteKeySource],
        delete_set: &EqDeleteSet,
        batch: Option<&RecordBatch>,
        num_rows: usize,
    ) -> Result<Vec<ArrayRef>> {
        sources
            .iter()
            .zip(delete_set.fields())
            .map(|(source, field)| match (source, batch) {
                (EqDeleteKeySource::Column(path), Some(batch)) => column_by_path(batch, path),
                (EqDeleteKeySource::Constant(literal), _) => {
                    create_primitive_array_repeated(&field.data_type, literal, num_rows)
                }
                (EqDeleteKeySource::ConstantInStruct { struct_path, value }, Some(batch)) => {
                    let struct_nulls = column_by_path(batch, struct_path)?.nulls().cloned();
                    let column =
                        create_primitive_array_repeated(&field.data_type, value, num_rows)?;
                    Ok(with_nulls(column, struct_nulls)?)
                }
                (
                    EqDeleteKeySource::Column(_) | EqDeleteKeySource::ConstantInStruct { .. },
                    None,
                ) => Err(Error::new(
                    ErrorKind::Unexpected,
                    "Equality delete key column requires a record batch",
                )),
            })
            .collect()
    }

    pub(super) fn get_selected_row_group_indices(
        predicate: &BoundPredicate,
        parquet_metadata: &Arc<ParquetMetaData>,
        field_id_map: &HashMap<i32, usize>,
        snapshot_schema: &Schema,
    ) -> Result<Vec<usize>> {
        let row_groups_metadata = parquet_metadata.row_groups();
        let mut results = Vec::with_capacity(row_groups_metadata.len());

        for (idx, row_group_metadata) in row_groups_metadata.iter().enumerate() {
            if RowGroupMetricsEvaluator::eval(
                predicate,
                row_group_metadata,
                field_id_map,
                snapshot_schema,
            )? {
                results.push(idx);
            }
        }

        Ok(results)
    }

    /// Computes a [`RowSelection`] by evaluating the filter predicate against
    /// the Parquet page index (column index + offset index).
    ///
    /// Returns `Ok(None)` when the Parquet file lacks column or offset index
    /// metadata (common with older files written before page indexes became
    /// standard). In that case page-level pruning is simply skipped; row-group
    /// filtering and the Arrow row filter still apply the predicate.
    ///
    /// `Ok(Some(empty))` case means that all rows were filtered by the predicate - returning zero rows
    pub(super) fn get_row_selection_for_filter_predicate(
        predicate: &BoundPredicate,
        parquet_metadata: &Arc<ParquetMetaData>,
        selected_row_groups: &Option<Vec<usize>>,
        field_id_map: &HashMap<i32, usize>,
        snapshot_schema: &Schema,
    ) -> Result<Option<RowSelection>> {
        let Some(column_index) = parquet_metadata.column_index() else {
            tracing::debug!("ColumnIndex was absent while reading this file");
            return Ok(None);
        };

        let Some(offset_index) = parquet_metadata.offset_index() else {
            tracing::debug!("OffsetIndex was absent while reading this file");
            return Ok(None);
        };

        // If all row groups were filtered out, return an empty RowSelection (select no rows)
        //
        if let Some(selected_row_groups) = selected_row_groups
            && selected_row_groups.is_empty()
        {
            return Ok(Some(RowSelection::from(Vec::new())));
        }

        let mut selected_row_groups_idx = 0;

        let page_index = column_index
            .iter()
            .enumerate()
            .zip(offset_index)
            .zip(parquet_metadata.row_groups());

        let mut results = Vec::new();
        for (((idx, column_index), offset_index), row_group_metadata) in page_index {
            if let Some(selected_row_groups) = selected_row_groups {
                // skip row groups that aren't present in selected_row_groups
                if idx == selected_row_groups[selected_row_groups_idx] {
                    selected_row_groups_idx += 1;
                } else {
                    continue;
                }
            }

            let selections_for_page = PageIndexEvaluator::eval(
                predicate,
                column_index,
                offset_index,
                row_group_metadata,
                field_id_map,
                snapshot_schema,
            )?;

            results.push(selections_for_page);

            if let Some(selected_row_groups) = selected_row_groups
                && selected_row_groups_idx == selected_row_groups.len()
            {
                break;
            }
        }

        Ok(Some(
            results.into_iter().flatten().collect::<Vec<_>>().into(),
        ))
    }

    /// Filters row groups by byte range to support Iceberg's file splitting.
    ///
    /// Engines split a data file into multiple scan tasks, each covering a byte range
    /// `[start, start+length)`. Normally Iceberg planning aligns these splits to row group
    /// boundaries using the data file's `split_offsets` metadata, so a task's range never
    /// bisects a row group. But when `split_offsets` is missing (e.g. a manually written or
    /// non-conforming file), planning falls back to tiling the file at the requested split
    /// size, and a task's range can land in the middle of a row group.
    ///
    /// A row group must be read by exactly one task, otherwise its rows are duplicated. We
    /// assign ownership by the row group's midpoint: a task owns a row group only if its range
    /// contains that midpoint. Because the tasks tile the file contiguously and disjointly,
    /// each midpoint falls in exactly one task. This matches parquet-mr's `BlockMetaData`
    /// midpoint semantics. For a whole-file task (`start=0, length=fileSize`, as iceberg-rust's
    /// own planner emits) every midpoint lies in range, so all row groups are selected.
    pub(super) fn filter_row_groups_by_byte_range(
        parquet_metadata: &Arc<ParquetMetaData>,
        start: u64,
        length: u64,
    ) -> Result<Vec<usize>> {
        let row_groups = parquet_metadata.row_groups();
        let mut selected = Vec::new();
        let end = start + length;

        // Row groups are stored sequentially after the 4-byte magic header.
        let mut current_byte_offset = 4u64;

        for (idx, row_group) in row_groups.iter().enumerate() {
            let row_group_size = row_group.compressed_size() as u64;
            let row_group_midpoint = current_byte_offset + row_group_size / 2;

            // Half-open ownership: a midpoint on a task boundary belongs to the upper task,
            // so exactly one task ever claims a given row group.
            if start <= row_group_midpoint && row_group_midpoint < end {
                selected.push(idx);
            }

            current_byte_offset += row_group_size;
        }

        Ok(selected)
    }
}

/// Where the values of an equality delete key column come from in a data file.
enum EqDeleteKeySource {
    /// Names from the batch root down to the Parquet leaf column.
    Column(Vec<String>),
    /// The file does not store the column, so every row reads this value (`None` is null).
    Constant(Option<PrimitiveLiteral>),
    /// The file does not store the column but stores an enclosing struct: rows read `value`
    /// where that struct is non-null, and null where it is null.
    ConstantInStruct {
        /// Names from the batch root down to the enclosing struct.
        struct_path: Vec<String>,
        value: Option<PrimitiveLiteral>,
    },
}

/// The structs enclosing `field_id`, outermost first, or `None` if `field_id` is not reached
/// through structs only.
fn struct_ancestors(r#struct: &StructType, field_id: i32) -> Option<Vec<NestedFieldRef>> {
    for field in r#struct.fields() {
        if field.id == field_id {
            return Some(Vec::new());
        }
        if let Type::Struct(child) = field.field_type.as_ref()
            && let Some(mut ancestors) = struct_ancestors(child, field_id)
        {
            ancestors.insert(0, field.clone());
            return Some(ancestors);
        }
    }
    None
}

/// The deepest of `ancestors` (structs, outermost first) that the file stores, looked up by
/// field id in the file's Arrow schema. Returns its depth, the names from the batch root down
/// to it, and its first Parquet leaf column: Arrow leaves follow the Parquet column order.
fn stored_struct(
    file_fields: &Fields,
    ancestors: &[NestedFieldRef],
) -> Option<(usize, Vec<String>, usize)> {
    let mut fields = file_fields;
    let mut first_leaf = 0;
    let mut path = Vec::with_capacity(ancestors.len());
    let mut stored = None;

    for (depth, ancestor) in ancestors.iter().enumerate() {
        let mut found = None;
        for field in fields {
            if arrow_field_id(field) == Some(ancestor.id) {
                found = Some(field);
                break;
            }
            first_leaf += leaf_count(field.data_type());
        }
        let Some(field) = found else { break };
        let DataType::Struct(children) = field.data_type() else {
            break;
        };
        path.push(field.name().clone());
        // A struct without stored leaves cannot be decoded, so its nulls are unknown.
        if leaf_count(field.data_type()) > 0 {
            stored = Some((depth, path.clone(), first_leaf));
        }
        fields = children;
    }
    stored
}

fn arrow_field_id(field: &Field) -> Option<i32> {
    field
        .metadata()
        .get(PARQUET_FIELD_ID_META_KEY)?
        .parse()
        .ok()
}

/// The number of Parquet leaf columns that store a value of `data_type`.
fn leaf_count(data_type: &DataType) -> usize {
    match data_type {
        DataType::Struct(fields) => fields.iter().map(|f| leaf_count(f.data_type())).sum(),
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::ListView(field)
        | DataType::LargeListView(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => leaf_count(field.data_type()),
        _ => 1,
    }
}

/// Extracts a (possibly struct-nested) leaf column from a `RowFilter` batch. A null parent
/// struct makes the leaf null.
fn column_by_path(batch: &RecordBatch, path: &[String]) -> Result<ArrayRef> {
    let not_found = || {
        Error::new(
            ErrorKind::DataInvalid,
            format!(
                "Equality delete key column `{}` not found in data file",
                path.join(".")
            ),
        )
    };

    let (root, rest) = path.split_first().ok_or_else(not_found)?;
    let mut column = batch.column_by_name(root).ok_or_else(not_found)?.clone();
    for name in rest {
        let struct_array = column.as_struct_opt().ok_or_else(not_found)?;
        let child = struct_array.column_by_name(name).ok_or_else(not_found)?;
        column = with_nulls(child.clone(), struct_array.nulls().cloned())?;
    }
    Ok(column)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fs::File;
    use std::sync::Arc;

    use arrow_array::cast::AsArray;
    use arrow_array::{
        ArrayRef, Int32Array, Int64Array, LargeStringArray, RecordBatch, StringArray,
    };
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use futures::TryStreamExt;
    use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
    use parquet::basic::Compression;
    use parquet::file::metadata::{FileMetaData, ParquetMetaData, ParquetMetaDataBuilder};
    use parquet::file::properties::{EnabledStatistics, WriterProperties};
    use parquet::schema::parser::parse_message_type;
    use parquet::schema::types::SchemaDescriptor;
    use tempfile::TempDir;

    use crate::Runtime;
    use crate::arrow::{ArrowReader, ArrowReaderBuilder};
    use crate::expr::{Bind, BoundPredicate, Predicate, Reference};
    use crate::io::FileIO;
    use crate::scan::{FileScanTask, FileScanTaskDeleteFile, FileScanTaskStream};
    use crate::spec::{
        DataContentType, DataFileFormat, Datum, NestedField, PrimitiveType, Schema, SchemaRef, Type,
    };

    async fn test_perform_read(
        predicate: Predicate,
        schema: SchemaRef,
        table_location: String,
        reader: ArrowReader,
    ) -> Vec<Option<String>> {
        let tasks = {
            let task = FileScanTask::builder()
                .with_file_size_in_bytes(
                    std::fs::metadata(format!("{table_location}/1.parquet"))
                        .unwrap()
                        .len(),
                )
                .with_start(0)
                .with_length(0)
                .with_data_file_path(format!("{table_location}/1.parquet"))
                .with_data_file_format(DataFileFormat::Parquet)
                .with_schema(schema.clone())
                .with_project_field_ids(vec![1])
                .with_predicate(Some(predicate.bind(schema, true).unwrap()))
                .with_case_sensitive(false)
                .build();
            Box::pin(futures::stream::iter(vec![Ok(task)])) as FileScanTaskStream
        };

        let result = reader
            .read(tasks)
            .unwrap()
            .stream()
            .try_collect::<Vec<RecordBatch>>()
            .await
            .unwrap();

        result[0].columns()[0]
            .as_string_opt::<i32>()
            .unwrap()
            .iter()
            .map(|v| v.map(ToOwned::to_owned))
            .collect::<Vec<_>>()
    }

    fn setup_kleene_logic(
        data_for_col_a: Vec<Option<String>>,
        col_a_type: DataType,
    ) -> (FileIO, SchemaRef, String, TempDir) {
        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::optional(1, "a", Type::Primitive(PrimitiveType::String)).into(),
                ])
                .build()
                .unwrap(),
        );

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("a", col_a_type.clone(), true).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
        ]));

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();

        let file_io = FileIO::new_with_fs();

        let col = match col_a_type {
            DataType::Utf8 => Arc::new(StringArray::from(data_for_col_a)) as ArrayRef,
            DataType::LargeUtf8 => Arc::new(LargeStringArray::from(data_for_col_a)) as ArrayRef,
            _ => panic!("unexpected col_a_type"),
        };

        let to_write = RecordBatch::try_new(arrow_schema.clone(), vec![col]).unwrap();

        // Write the Parquet files
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();

        let file = File::create(format!("{table_location}/1.parquet")).unwrap();
        let mut writer =
            ArrowWriter::try_new(file, to_write.schema(), Some(props.clone())).unwrap();

        writer.write(&to_write).expect("Writing batch");

        // writer must be closed to write footer
        writer.close().unwrap();

        (file_io, schema, table_location, tmp_dir)
    }

    #[tokio::test]
    async fn test_kleene_logic_or_behaviour() {
        // a IS NULL OR a = 'foo'
        let predicate = Reference::new("a")
            .is_null()
            .or(Reference::new("a").equal_to(Datum::string("foo")));

        // Table data: [NULL, "foo", "bar"]
        let data_for_col_a = vec![None, Some("foo".to_string()), Some("bar".to_string())];

        // Expected: [NULL, "foo"].
        let expected = vec![None, Some("foo".to_string())];

        let (file_io, schema, table_location, _temp_dir) =
            setup_kleene_logic(data_for_col_a, DataType::Utf8);
        let reader = ArrowReaderBuilder::new(file_io, Runtime::current()).build();

        let result_data = test_perform_read(predicate, schema, table_location, reader).await;

        assert_eq!(result_data, expected);
    }

    #[tokio::test]
    async fn test_kleene_logic_and_behaviour() {
        // a IS NOT NULL AND a != 'foo'
        let predicate = Reference::new("a")
            .is_not_null()
            .and(Reference::new("a").not_equal_to(Datum::string("foo")));

        // Table data: [NULL, "foo", "bar"]
        let data_for_col_a = vec![None, Some("foo".to_string()), Some("bar".to_string())];

        // Expected: ["bar"].
        let expected = vec![Some("bar".to_string())];

        let (file_io, schema, table_location, _temp_dir) =
            setup_kleene_logic(data_for_col_a, DataType::Utf8);
        let reader = ArrowReaderBuilder::new(file_io, Runtime::current()).build();

        let result_data = test_perform_read(predicate, schema, table_location, reader).await;

        assert_eq!(result_data, expected);
    }

    #[tokio::test]
    async fn test_predicate_cast_literal() {
        let predicates = vec![
            // a == 'foo'
            (Reference::new("a").equal_to(Datum::string("foo")), vec![
                Some("foo".to_string()),
            ]),
            // a != 'foo'
            (
                Reference::new("a").not_equal_to(Datum::string("foo")),
                vec![Some("bar".to_string())],
            ),
            // STARTS_WITH(a, 'foo')
            (Reference::new("a").starts_with(Datum::string("f")), vec![
                Some("foo".to_string()),
            ]),
            // NOT STARTS_WITH(a, 'foo')
            (
                Reference::new("a").not_starts_with(Datum::string("f")),
                vec![Some("bar".to_string())],
            ),
            // a < 'foo'
            (Reference::new("a").less_than(Datum::string("foo")), vec![
                Some("bar".to_string()),
            ]),
            // a <= 'foo'
            (
                Reference::new("a").less_than_or_equal_to(Datum::string("foo")),
                vec![Some("foo".to_string()), Some("bar".to_string())],
            ),
            // a > 'foo'
            (
                Reference::new("a").greater_than(Datum::string("bar")),
                vec![Some("foo".to_string())],
            ),
            // a >= 'foo'
            (
                Reference::new("a").greater_than_or_equal_to(Datum::string("foo")),
                vec![Some("foo".to_string())],
            ),
            // a IN ('foo', 'bar')
            (
                Reference::new("a").is_in([Datum::string("foo"), Datum::string("baz")]),
                vec![Some("foo".to_string())],
            ),
            // a NOT IN ('foo', 'bar')
            (
                Reference::new("a").is_not_in([Datum::string("foo"), Datum::string("baz")]),
                vec![Some("bar".to_string())],
            ),
        ];

        // Table data: ["foo", "bar"]
        let data_for_col_a = vec![Some("foo".to_string()), Some("bar".to_string())];

        let (file_io, schema, table_location, _temp_dir) =
            setup_kleene_logic(data_for_col_a, DataType::LargeUtf8);
        let reader = ArrowReaderBuilder::new(file_io, Runtime::current()).build();

        for (predicate, expected) in predicates {
            println!("testing predicate {predicate}");
            let result_data = test_perform_read(
                predicate.clone(),
                schema.clone(),
                table_location.clone(),
                reader.clone(),
            )
            .await;

            assert_eq!(result_data, expected, "predicate={predicate}");
        }
    }

    /// Verifies that file splits respect byte ranges and only read specific row groups.
    #[tokio::test]
    async fn test_file_splits_respect_byte_ranges() {
        use arrow_array::Int32Array;
        use parquet::file::reader::{FileReader, SerializedFileReader};

        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                ])
                .build()
                .unwrap(),
        );

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
        ]));

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();
        let file_path = format!("{table_location}/multi_row_group.parquet");

        // Force each batch into its own row group for testing byte range filtering.
        let batch1 = RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(Int32Array::from(
            (0..100).collect::<Vec<i32>>(),
        ))])
        .unwrap();
        let batch2 = RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(Int32Array::from(
            (100..200).collect::<Vec<i32>>(),
        ))])
        .unwrap();
        let batch3 = RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(Int32Array::from(
            (200..300).collect::<Vec<i32>>(),
        ))])
        .unwrap();

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .set_max_row_group_row_count(Some(100))
            .build();

        let file = File::create(&file_path).unwrap();
        let mut writer = ArrowWriter::try_new(file, arrow_schema.clone(), Some(props)).unwrap();
        writer.write(&batch1).expect("Writing batch 1");
        writer.write(&batch2).expect("Writing batch 2");
        writer.write(&batch3).expect("Writing batch 3");
        writer.close().unwrap();

        // Read the file metadata to get row group byte positions
        let file = File::open(&file_path).unwrap();
        let reader = SerializedFileReader::new(file).unwrap();
        let metadata = reader.metadata();

        println!("File has {} row groups", metadata.num_row_groups());
        assert_eq!(metadata.num_row_groups(), 3, "Expected 3 row groups");

        // Get byte positions for each row group
        let row_group_0 = metadata.row_group(0);
        let row_group_1 = metadata.row_group(1);
        let row_group_2 = metadata.row_group(2);

        let rg0_start = 4u64; // Parquet files start with 4-byte magic "PAR1"
        let rg1_start = rg0_start + row_group_0.compressed_size() as u64;
        let rg2_start = rg1_start + row_group_1.compressed_size() as u64;
        let file_end = rg2_start + row_group_2.compressed_size() as u64;

        println!(
            "Row group 0: {} rows, starts at byte {}, {} bytes compressed",
            row_group_0.num_rows(),
            rg0_start,
            row_group_0.compressed_size()
        );
        println!(
            "Row group 1: {} rows, starts at byte {}, {} bytes compressed",
            row_group_1.num_rows(),
            rg1_start,
            row_group_1.compressed_size()
        );
        println!(
            "Row group 2: {} rows, starts at byte {}, {} bytes compressed",
            row_group_2.num_rows(),
            rg2_start,
            row_group_2.compressed_size()
        );

        let file_io = FileIO::new_with_fs();
        let reader = ArrowReaderBuilder::new(file_io, Runtime::current()).build();

        // Task 1: read only the first row group
        let task1 = FileScanTask::builder()
            .with_file_size_in_bytes(std::fs::metadata(&file_path).unwrap().len())
            .with_start(rg0_start)
            .with_length(row_group_0.compressed_size() as u64)
            .with_data_file_path(file_path.clone())
            .with_data_file_format(DataFileFormat::Parquet)
            .with_schema(schema.clone())
            .with_project_field_ids(vec![1])
            .with_record_count(Some(100))
            .with_case_sensitive(false)
            .build();

        // Task 2: read the second and third row groups
        let task2 = FileScanTask::builder()
            .with_file_size_in_bytes(std::fs::metadata(&file_path).unwrap().len())
            .with_start(rg1_start)
            .with_length(file_end - rg1_start)
            .with_data_file_path(file_path.clone())
            .with_data_file_format(DataFileFormat::Parquet)
            .with_schema(schema.clone())
            .with_project_field_ids(vec![1])
            .with_record_count(Some(200))
            .with_case_sensitive(false)
            .build();

        let tasks1 = Box::pin(futures::stream::iter(vec![Ok(task1)])) as FileScanTaskStream;
        let result1 = reader
            .clone()
            .read(tasks1)
            .unwrap()
            .stream()
            .try_collect::<Vec<RecordBatch>>()
            .await
            .unwrap();

        let total_rows_task1: usize = result1.iter().map(|b| b.num_rows()).sum();
        println!(
            "Task 1 (bytes {}-{}) returned {} rows",
            rg0_start,
            rg0_start + row_group_0.compressed_size() as u64,
            total_rows_task1
        );

        let tasks2 = Box::pin(futures::stream::iter(vec![Ok(task2)])) as FileScanTaskStream;
        let result2 = reader
            .read(tasks2)
            .unwrap()
            .stream()
            .try_collect::<Vec<RecordBatch>>()
            .await
            .unwrap();

        let total_rows_task2: usize = result2.iter().map(|b| b.num_rows()).sum();
        println!("Task 2 (bytes {rg1_start}-{file_end}) returned {total_rows_task2} rows");

        assert_eq!(
            total_rows_task1, 100,
            "Task 1 should read only the first row group (100 rows), but got {total_rows_task1} rows"
        );

        assert_eq!(
            total_rows_task2, 200,
            "Task 2 should read only the second+third row groups (200 rows), but got {total_rows_task2} rows"
        );

        // Verify the actual data values are correct (not just the row count)
        if total_rows_task1 > 0 {
            let first_batch = &result1[0];
            let id_col = first_batch
                .column(0)
                .as_primitive::<arrow_array::types::Int32Type>();
            let first_val = id_col.value(0);
            let last_val = id_col.value(id_col.len() - 1);
            println!("Task 1 data range: {first_val} to {last_val}");

            assert_eq!(first_val, 0, "Task 1 should start with id=0");
            assert_eq!(last_val, 99, "Task 1 should end with id=99");
        }

        if total_rows_task2 > 0 {
            let first_batch = &result2[0];
            let id_col = first_batch
                .column(0)
                .as_primitive::<arrow_array::types::Int32Type>();
            let first_val = id_col.value(0);
            println!("Task 2 first value: {first_val}");

            assert_eq!(first_val, 100, "Task 2 should start with id=100, not id=0");
        }
    }

    /// A single data file split into multiple sub-row-group byte ranges (as Spark/Iceberg
    /// planning produces when split-size is smaller than a row group) must still yield each
    /// row exactly once. The previous overlap-based selection let every split whose byte range
    /// touched a row group read it, duplicating rows; ownership by midpoint reads each row group
    /// from exactly one split.
    #[tokio::test]
    async fn test_sub_row_group_splits_do_not_duplicate_rows() {
        use arrow_array::Int32Array;

        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                ])
                .build()
                .unwrap(),
        );

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
        ]));

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();
        let file_path = format!("{table_location}/sub_split.parquet");

        // Three row groups of 100 rows each (ids 0..300).
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .set_max_row_group_row_count(Some(100))
            .build();
        let file = File::create(&file_path).unwrap();
        let mut writer = ArrowWriter::try_new(file, arrow_schema.clone(), Some(props)).unwrap();
        for chunk in [0..100, 100..200, 200..300] {
            let batch = RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(
                Int32Array::from(chunk.collect::<Vec<i32>>()),
            )])
            .unwrap();
            writer.write(&batch).expect("Writing batch");
        }
        writer.close().unwrap();

        let file_size = std::fs::metadata(&file_path).unwrap().len();
        let file_io = FileIO::new_with_fs();
        let reader = ArrowReaderBuilder::new(file_io, Runtime::current()).build();

        // Tile the whole file into 64-byte splits, mirroring Spark's split-size planning, and
        // read every split. A 64-byte split is far smaller than a row group, so each row group
        // is touched by several splits but must be owned (read) by exactly one.
        let mut ids = Vec::new();
        let split_size = 64u64;
        let mut start = 0u64;
        while start < file_size {
            let length = split_size.min(file_size - start);
            let task = FileScanTask::builder()
                .with_file_size_in_bytes(file_size)
                .with_start(start)
                .with_length(length)
                .with_data_file_path(file_path.clone())
                .with_data_file_format(DataFileFormat::Parquet)
                .with_schema(schema.clone())
                .with_project_field_ids(vec![1])
                .with_case_sensitive(false)
                .build();

            let tasks = Box::pin(futures::stream::iter(vec![Ok(task)])) as FileScanTaskStream;
            let batches = reader
                .clone()
                .read(tasks)
                .unwrap()
                .stream()
                .try_collect::<Vec<RecordBatch>>()
                .await
                .unwrap();

            for batch in &batches {
                let col = batch
                    .column(0)
                    .as_primitive::<arrow_array::types::Int32Type>();
                ids.extend(col.values().iter().copied());
            }

            start += length;
        }

        ids.sort_unstable();
        assert_eq!(
            ids,
            (0..300).collect::<Vec<i32>>(),
            "each row must be read exactly once across all splits, got {} rows",
            ids.len()
        );
    }

    /// When a split boundary lands exactly on a row group's midpoint, half-open ownership
    /// (`start <= midpoint < end`) must hand that row group to the upper split only: the lower
    /// split ends at the midpoint and so excludes it, the upper split starts at the midpoint and
    /// so claims it. Two splits meeting exactly at the middle row group's midpoint must therefore
    /// read every row once, with the middle row group going to the upper split.
    #[tokio::test]
    async fn test_split_boundary_on_row_group_midpoint() {
        use arrow_array::Int32Array;
        use parquet::file::reader::{FileReader, SerializedFileReader};

        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                ])
                .build()
                .unwrap(),
        );

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
        ]));

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();
        let file_path = format!("{table_location}/midpoint.parquet");

        // Three row groups of 100 rows each (ids 0..300).
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .set_max_row_group_row_count(Some(100))
            .build();
        let file = File::create(&file_path).unwrap();
        let mut writer = ArrowWriter::try_new(file, arrow_schema.clone(), Some(props)).unwrap();
        for chunk in [0..100, 100..200, 200..300] {
            let batch = RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(
                Int32Array::from(chunk.collect::<Vec<i32>>()),
            )])
            .unwrap();
            writer.write(&batch).expect("Writing batch");
        }
        writer.close().unwrap();

        // Locate the middle row group's exact midpoint. Row groups are stored back to back after
        // the 4-byte magic header.
        let metadata = SerializedFileReader::new(File::open(&file_path).unwrap())
            .unwrap()
            .metadata()
            .clone();
        assert_eq!(metadata.num_row_groups(), 3);
        let rg1_start = 4 + metadata.row_group(0).compressed_size() as u64;
        let rg1_size = metadata.row_group(1).compressed_size() as u64;
        let rg1_midpoint = rg1_start + rg1_size / 2;
        let file_end = rg1_start + rg1_size + metadata.row_group(2).compressed_size() as u64;

        let file_size = std::fs::metadata(&file_path).unwrap().len();
        let file_io = FileIO::new_with_fs();
        let reader = ArrowReaderBuilder::new(file_io, Runtime::current()).build();

        // Two splits meeting exactly at rg1's midpoint. The lower split ends there, the upper
        // starts there; the middle row group must fall to the upper split alone.
        let mut per_split = Vec::new();
        for (start, end) in [(0, rg1_midpoint), (rg1_midpoint, file_end)] {
            let task = FileScanTask::builder()
                .with_file_size_in_bytes(file_size)
                .with_start(start)
                .with_length(end - start)
                .with_data_file_path(file_path.clone())
                .with_data_file_format(DataFileFormat::Parquet)
                .with_schema(schema.clone())
                .with_project_field_ids(vec![1])
                .with_case_sensitive(false)
                .build();

            let tasks = Box::pin(futures::stream::iter(vec![Ok(task)])) as FileScanTaskStream;
            let batches = reader
                .clone()
                .read(tasks)
                .unwrap()
                .stream()
                .try_collect::<Vec<RecordBatch>>()
                .await
                .unwrap();

            let mut ids = Vec::new();
            for batch in &batches {
                let col = batch
                    .column(0)
                    .as_primitive::<arrow_array::types::Int32Type>();
                ids.extend(col.values().iter().copied());
            }
            per_split.push(ids);
        }

        assert_eq!(
            per_split[0],
            (0..100).collect::<Vec<i32>>(),
            "lower split, ending at rg1's midpoint, must read only rg0"
        );
        assert_eq!(
            per_split[1],
            (100..300).collect::<Vec<i32>>(),
            "upper split, starting at rg1's midpoint, must read rg1 and rg2"
        );
    }

    fn int_schema() -> SchemaRef {
        Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![Arc::new(NestedField::required(
                    1,
                    "x",
                    Type::Primitive(PrimitiveType::Int),
                ))])
                .build()
                .unwrap(),
        )
    }

    fn simple_predicate(schema: SchemaRef) -> BoundPredicate {
        Reference::new("x")
            .greater_than(Datum::int(0))
            .bind(schema.clone(), false)
            .unwrap()
    }

    fn field_id_map() -> HashMap<i32, usize> {
        let mut m = HashMap::new();
        m.insert(1_i32, 0_usize);
        m
    }

    fn metadata_no_page_indexes() -> Arc<ParquetMetaData> {
        let msg_type = parse_message_type("message schema { REQUIRED INT32 x; }").unwrap();
        let schema_desc = Arc::new(SchemaDescriptor::new(Arc::new(msg_type)));
        let file_meta = FileMetaData::new(2, 0, None, None, schema_desc.clone(), None);
        Arc::new(ParquetMetaDataBuilder::new(file_meta).build())
    }

    fn metadata_column_index_only() -> Arc<ParquetMetaData> {
        let msg_type = parse_message_type("message schema { REQUIRED INT32 x; }").unwrap();
        let schema_desc = Arc::new(SchemaDescriptor::new(Arc::new(msg_type)));
        let file_meta = FileMetaData::new(2, 0, None, None, schema_desc, None);
        Arc::new(
            ParquetMetaDataBuilder::new(file_meta)
                .set_column_index(Some(vec![]))
                .build(),
        )
    }

    fn metadata_with_both_indexes() -> Arc<ParquetMetaData> {
        let msg_type = parse_message_type("message schema { REQUIRED INT32 x; }").unwrap();
        let schema_desc = Arc::new(SchemaDescriptor::new(Arc::new(msg_type)));
        let file_meta = FileMetaData::new(2, 0, None, None, schema_desc, None);
        Arc::new(
            ParquetMetaDataBuilder::new(file_meta)
                .set_column_index(Some(vec![]))
                .set_offset_index(Some(vec![]))
                .build(),
        )
    }

    /// Testing suite regarding: https://github.com/apache/iceberg-rust/issues/2452
    /// Testing when: both indices are absent, some present, both present
    #[test]
    fn test_absent_column_index_returns_ok_none() {
        let schema = int_schema();
        let predicate = simple_predicate(schema.clone());
        let metadata = metadata_no_page_indexes();
        let field_id_map = field_id_map();

        let result = ArrowReader::get_row_selection_for_filter_predicate(
            &predicate,
            &metadata,
            &None,
            &field_id_map,
            schema.as_ref(),
        );

        assert!(
            result.is_ok(),
            "expected Ok(_), got Err: {:?}",
            result.unwrap_err()
        );
        assert!(
            result.unwrap().is_none(),
            "expected Ok(None) when column index is absent"
        );
    }

    #[test]
    fn test_absent_offset_index_returns_ok_none() {
        let schema = int_schema();
        let predicate = simple_predicate(schema.clone());
        let metadata = metadata_column_index_only();
        let field_id_map = field_id_map();

        let result = ArrowReader::get_row_selection_for_filter_predicate(
            &predicate,
            &metadata,
            &None,
            &field_id_map,
            schema.as_ref(),
        );

        assert!(
            result.is_ok(),
            "expected Ok(_), got Err: {:?}",
            result.unwrap_err()
        );
        assert!(
            result.unwrap().is_none(),
            "expected Ok(None) when offset index is absent"
        );
    }

    #[test]
    fn test_absent_column_index_with_selected_row_groups_returns_ok_none() {
        let schema = int_schema();
        let predicate = simple_predicate(schema.clone());
        let metadata = metadata_no_page_indexes();
        let field_id_map = field_id_map();
        let selected = Some(vec![0usize, 1]);

        let result = ArrowReader::get_row_selection_for_filter_predicate(
            &predicate,
            &metadata,
            &selected,
            &field_id_map,
            schema.as_ref(),
        );

        assert!(result.is_ok());
        assert!(
            result.unwrap().is_none(),
            "absent column index must short-circuit before selected_row_groups is inspected"
        );
    }

    #[test]
    fn test_absent_offset_index_with_selected_row_groups_returns_ok_none() {
        let schema = int_schema();
        let predicate = simple_predicate(schema.clone());
        let metadata = metadata_column_index_only();
        let field_id_map = field_id_map();
        let selected = Some(vec![0usize]);

        let result = ArrowReader::get_row_selection_for_filter_predicate(
            &predicate,
            &metadata,
            &selected,
            &field_id_map,
            schema.as_ref(),
        );

        assert!(result.is_ok());
        assert!(
            result.unwrap().is_none(),
            "absent offset index must short-circuit before selected_row_groups is inspected"
        );
    }

    #[test]
    fn test_absent_column_index_with_empty_selected_row_groups_returns_ok_none() {
        let schema = int_schema();
        let predicate = simple_predicate(schema.clone());
        let metadata = metadata_no_page_indexes();
        let field_id_map = field_id_map();
        let selected: Option<Vec<usize>> = Some(vec![]);

        let result = ArrowReader::get_row_selection_for_filter_predicate(
            &predicate,
            &metadata,
            &selected,
            &field_id_map,
            schema.as_ref(),
        );

        assert!(result.is_ok());
        assert!(
            result.unwrap().is_none(),
            "column index check must fire before the empty-selected-row-groups branch"
        );
    }

    #[test]
    fn test_both_indexes_present_empty_selected_row_groups_returns_ok_some_empty() {
        let schema = int_schema();
        let predicate = simple_predicate(schema.clone());
        let metadata = metadata_with_both_indexes();
        let field_id_map = field_id_map();
        let selected: Option<Vec<usize>> = Some(vec![]);

        let result = ArrowReader::get_row_selection_for_filter_predicate(
            &predicate,
            &metadata,
            &selected,
            &field_id_map,
            schema.as_ref(),
        );

        assert!(
            result.is_ok(),
            "expected Ok(_), got Err: {:?}",
            result.unwrap_err()
        );

        let row_selection = result.unwrap().expect(
            "expected Ok(Some(_)) when both indexes are present and all row groups are filtered",
        );

        assert_eq!(
            row_selection.row_count(),
            0,
            "RowSelection must be empty (zero rows selected) when selected_row_groups is empty"
        );
    }

    /// Full-suite regression test for issue: https://github.com/apache/iceberg-rust/issues/2452
    #[tokio::test]
    async fn test_scan_without_page_indexes_does_not_error() {
        // Building schema (both iceberg and arrow for RecordBatch)
        let iceberg_schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                ])
                .build()
                .unwrap(),
        );

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
            Field::new("name", DataType::Utf8, true).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "2".to_string(),
            )])),
        ]));

        let tmp_dir = TempDir::new().unwrap();
        let file_path = format!("{}/data.parquet", tmp_dir.path().to_str().unwrap());

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            // Disabling page statistics
            .set_statistics_enabled(EnabledStatistics::None)
            .build();

        let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])) as ArrayRef,
            Arc::new(StringArray::from(vec![
                Some("alice"),
                Some("bob"),
                None,
                Some("dana"),
                Some("eve"),
            ])) as ArrayRef,
        ])
        .unwrap();

        let file = File::create(&file_path).unwrap();
        let mut writer = ArrowWriter::try_new(file, arrow_schema.clone(), Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        // Truly exercising a file without column/offset index
        {
            use parquet::file::reader::{FileReader, SerializedFileReader};
            let f = File::open(&file_path).unwrap();
            let rdr = SerializedFileReader::new(f).unwrap();
            assert!(
                rdr.metadata().column_index().is_none(),
                "test fixture must produce a file without a column index"
            );
            assert!(
                rdr.metadata().offset_index().is_none(),
                "test fixture must a product a file without offset index"
            )
        }

        // Predicate: id > 2
        let predicate = Reference::new("id")
            .greater_than(Datum::int(2))
            .bind(iceberg_schema.clone(), false)
            .unwrap();

        let file_io = FileIO::new_with_fs();
        let reader = ArrowReaderBuilder::new(file_io.clone(), Runtime::current())
            // Enabling row selection ()
            .with_row_selection_enabled(true)
            .build();

        let file_size = std::fs::metadata(&file_path).unwrap().len();
        let task = FileScanTask {
            file_size_in_bytes: file_size,
            start: 0,
            length: 0,
            record_count: None,
            first_row_id: None,
            data_sequence_number: None,
            data_file_path: file_path.clone(),
            data_file_format: DataFileFormat::Parquet,
            schema: iceberg_schema.clone(),
            project_field_ids: vec![1, 2],
            predicate: Some(predicate),
            deletes: vec![],
            sequence_number: 0,
            file_sequence_number: None,
            partition: None,
            partition_spec: None,
            name_mapping: None,
            unified_partition_type: None,
            case_sensitive: false,
            key_metadata: None,
        };

        let stream = Box::pin(futures::stream::iter(vec![Ok(task)])) as FileScanTaskStream;
        let batches: Vec<RecordBatch> = reader
            .read(stream)
            .unwrap()
            .stream()
            .try_collect()
            .await
            .unwrap();

        let ids: Vec<i32> = batches
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .values()
                    .iter()
                    .copied()
            })
            .collect();

        assert_eq!(
            ids,
            vec![3, 4, 5],
            "predicate must still be enforced via Arrow row filter even without page indexes"
        );

        // Absent index + position delete field present
        let pos_del_path = format!("{}/pos-del.parquet", tmp_dir.path().to_str().unwrap());

        // Build the position delete Arrow schema (standard Iceberg layout).
        let pos_del_arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("file_path", DataType::Utf8, false),
            Field::new("pos", DataType::Int64, false),
        ]));

        let pos_del_batch = RecordBatch::try_new(pos_del_arrow_schema.clone(), vec![
            // Both deletions reference the same data file
            Arc::new(StringArray::from(vec![
                file_path.as_str(),
                file_path.as_str(),
            ])) as ArrayRef,
            // Delete by index - index-0 (`1` in test case) and index-2 (`3` in test case)
            Arc::new(Int64Array::from(vec![0i64, 2i64])) as ArrayRef,
        ])
        .unwrap();

        // Write position delete file also without indices
        let pos_del_props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .set_statistics_enabled(EnabledStatistics::None)
            .build();

        let pos_del_file = File::create(&pos_del_path).unwrap();
        let mut pos_del_writer = ArrowWriter::try_new(
            pos_del_file,
            pos_del_arrow_schema.clone(),
            Some(pos_del_props),
        )
        .unwrap();
        pos_del_writer.write(&pos_del_batch).unwrap();
        pos_del_writer.close().unwrap();

        // Predicate: id > 1 (note that `3` was also removed by position delete)
        let predicate_sub2 = Reference::new("id")
            .greater_than(Datum::int(1))
            .bind(iceberg_schema.clone(), false)
            .unwrap();

        let reader_sub2 = ArrowReaderBuilder::new(file_io.clone(), Runtime::current())
            .with_row_selection_enabled(true)
            .build();

        let task_sub2 = FileScanTask {
            file_size_in_bytes: file_size,
            start: 0,
            length: 0,
            record_count: None,
            first_row_id: None,
            data_sequence_number: None,
            data_file_path: file_path.clone(),
            data_file_format: DataFileFormat::Parquet,
            schema: iceberg_schema.clone(),
            project_field_ids: vec![1, 2],
            predicate: Some(predicate_sub2),
            deletes: vec![FileScanTaskDeleteFile {
                file_path: pos_del_path.clone(),
                file_type: DataContentType::PositionDeletes,
                partition_spec_id: 0,
                equality_ids: None,
                file_size_in_bytes: std::fs::metadata(&pos_del_path).unwrap().len(),
                file_format: DataFileFormat::Parquet,
                referenced_data_file: None,
                content_offset: None,
                content_size_in_bytes: None,
                record_count: None,
                sequence_number: 0,
                key_metadata: None,
            }],
            sequence_number: 0,
            file_sequence_number: None,
            partition: None,
            partition_spec: None,
            name_mapping: None,
            unified_partition_type: None,
            case_sensitive: false,
            key_metadata: None,
        };

        let stream_sub2 =
            Box::pin(futures::stream::iter(vec![Ok(task_sub2)])) as FileScanTaskStream;
        let batches_sub2: Vec<RecordBatch> = reader_sub2
            .read(stream_sub2)
            .unwrap()
            .stream()
            .try_collect()
            .await
            .unwrap();

        let ids_sub2: Vec<i32> = batches_sub2
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .values()
                    .iter()
                    .copied()
            })
            .collect();

        assert_eq!(
            ids_sub2,
            vec![2, 4, 5],
            "positional deletes must be applied correctly even when page indexes are absent"
        );
    }
}

#[cfg(test)]
mod equality_delete_tests {
    use std::collections::HashMap;
    use std::fs::File;
    use std::sync::Arc;

    use arrow_array::cast::AsArray;
    use arrow_array::types::Int64Type;
    use arrow_array::{ArrayRef, Int32Array, Int64Array, RecordBatch, StringArray, StructArray};
    use arrow_schema::{DataType, Field, Fields, Schema as ArrowSchema};
    use futures::TryStreamExt;
    use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
    use tempfile::TempDir;

    use crate::Runtime;
    use crate::arrow::ArrowReaderBuilder;
    use crate::expr::{Bind, Reference};
    use crate::io::FileIO;
    use crate::scan::{FileScanTask, FileScanTaskDeleteFile, FileScanTaskStream};
    use crate::spec::{
        DataContentType, DataFileFormat, Datum, Literal, NestedField, PrimitiveType, Schema,
        SchemaRef, StructType, Type,
    };

    fn field(name: &str, data_type: DataType, field_id: i32) -> Field {
        Field::new(name, data_type, true).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            field_id.to_string(),
        )]))
    }

    fn write_parquet(dir: &TempDir, name: &str, columns: Vec<(Field, ArrayRef)>) -> String {
        let (fields, arrays): (Vec<_>, Vec<_>) = columns.into_iter().unzip();
        let batch = RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), arrays).unwrap();
        let path = format!("{}/{name}", dir.path().to_str().unwrap());
        let mut writer =
            ArrowWriter::try_new(File::create(&path).unwrap(), batch.schema(), None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        path
    }

    fn eq_delete(path: &str, equality_ids: Vec<i32>) -> FileScanTaskDeleteFile {
        FileScanTaskDeleteFile::builder()
            .with_file_path(path.to_string())
            .with_file_size_in_bytes(std::fs::metadata(path).unwrap().len())
            .with_file_type(DataContentType::EqualityDeletes)
            .with_partition_spec_id(0)
            .with_equality_ids(Some(equality_ids))
            .build()
    }

    /// Reads `data_path` with the given deletes and returns the `id` column (field id 1).
    async fn read_ids(
        schema: SchemaRef,
        data_path: &str,
        deletes: Vec<FileScanTaskDeleteFile>,
        predicate: Option<crate::expr::Predicate>,
    ) -> Vec<i64> {
        let task = FileScanTask::builder()
            .with_file_size_in_bytes(std::fs::metadata(data_path).unwrap().len())
            .with_start(0)
            .with_length(0)
            .with_data_file_path(data_path.to_string())
            .with_data_file_format(DataFileFormat::Parquet)
            .with_schema(schema.clone())
            .with_project_field_ids(vec![1])
            .with_predicate(predicate.map(|p| p.bind(schema, true).unwrap()))
            .with_deletes(deletes)
            .with_case_sensitive(true)
            .build();

        let reader = ArrowReaderBuilder::new(FileIO::new_with_fs(), Runtime::current()).build();
        let tasks = Box::pin(futures::stream::iter(vec![Ok(task)])) as FileScanTaskStream;
        let batches: Vec<RecordBatch> = reader
            .read(tasks)
            .unwrap()
            .stream()
            .try_collect()
            .await
            .unwrap();

        batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column(0)
                    .as_primitive::<Int64Type>()
                    .values()
                    .to_vec()
            })
            .collect()
    }

    fn id_name_schema() -> SchemaRef {
        Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                    NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                ])
                .build()
                .unwrap(),
        )
    }

    // Delete files on different key columns apply independently, a null delete value removes
    // only null rows, and the scan predicate still applies alongside the deletes.
    #[tokio::test]
    async fn test_eq_deletes_with_scan_predicate() {
        let dir = TempDir::new().unwrap();
        let data = write_parquet(&dir, "data.parquet", vec![
            (
                field("id", DataType::Int64, 1),
                Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5, 6])) as ArrayRef,
            ),
            (
                field("name", DataType::Utf8, 2),
                Arc::new(StringArray::from(vec![
                    Some("a"),
                    None,
                    Some("c"),
                    Some("d"),
                    Some("e"),
                    Some("f"),
                ])) as ArrayRef,
            ),
        ]);
        let id_deletes = write_parquet(&dir, "id-deletes.parquet", vec![(
            field("id", DataType::Int64, 1),
            Arc::new(Int64Array::from(vec![4, 100])) as ArrayRef,
        )]);
        let name_deletes = write_parquet(&dir, "name-deletes.parquet", vec![(
            field("name", DataType::Utf8, 2),
            Arc::new(StringArray::from(vec![Some("e"), None])) as ArrayRef,
        )]);

        let ids = read_ids(
            id_name_schema(),
            &data,
            vec![
                eq_delete(&id_deletes, vec![1]),
                eq_delete(&name_deletes, vec![2]),
            ],
            Some(Reference::new("id").greater_than(Datum::long(1))),
        )
        .await;
        assert_eq!(ids, vec![3, 6]);
    }

    // Two delete files on the same key are merged; a composite key deletes a row only when
    // every column matches.
    #[tokio::test]
    async fn test_eq_deletes_merged_and_composite_key() {
        let dir = TempDir::new().unwrap();
        let data = write_parquet(&dir, "data.parquet", vec![
            (
                field("id", DataType::Int64, 1),
                Arc::new(Int64Array::from(vec![1, 2, 3, 4])) as ArrayRef,
            ),
            (
                field("name", DataType::Utf8, 2),
                Arc::new(StringArray::from(vec!["a", "b", "c", "d"])) as ArrayRef,
            ),
        ]);
        let first = write_parquet(&dir, "first.parquet", vec![(
            field("id", DataType::Int64, 1),
            Arc::new(Int64Array::from(vec![1])) as ArrayRef,
        )]);
        let second = write_parquet(&dir, "second.parquet", vec![(
            field("id", DataType::Int64, 1),
            Arc::new(Int64Array::from(vec![2])) as ArrayRef,
        )]);
        let composite = write_parquet(&dir, "composite.parquet", vec![
            (
                field("id", DataType::Int64, 1),
                Arc::new(Int64Array::from(vec![3, 4])) as ArrayRef,
            ),
            (
                field("name", DataType::Utf8, 2),
                Arc::new(StringArray::from(vec!["c", "x"])) as ArrayRef,
            ),
        ]);

        let ids = read_ids(
            id_name_schema(),
            &data,
            vec![
                eq_delete(&first, vec![1]),
                eq_delete(&composite, vec![1, 2]),
                eq_delete(&second, vec![1]),
            ],
            None,
        )
        .await;
        assert_eq!(ids, vec![4]);
    }

    // A data file written before an int -> long promotion still stores int32 keys.
    #[tokio::test]
    async fn test_eq_delete_on_promoted_column() {
        let dir = TempDir::new().unwrap();
        let data = write_parquet(&dir, "data.parquet", vec![(
            field("id", DataType::Int32, 1),
            Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef,
        )]);
        let deletes = write_parquet(&dir, "deletes.parquet", vec![(
            field("id", DataType::Int64, 1),
            Arc::new(Int64Array::from(vec![2])) as ArrayRef,
        )]);

        let ids = read_ids(
            id_name_schema(),
            &data,
            vec![eq_delete(&deletes, vec![1])],
            None,
        )
        .await;
        assert_eq!(ids, vec![1, 3]);
    }

    // A key column the data file does not store reads as its `initial-default` for every row.
    #[tokio::test]
    async fn test_eq_delete_on_missing_column_uses_initial_default() {
        let dir = TempDir::new().unwrap();
        let schema = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                    NestedField::optional(3, "b", Type::Primitive(PrimitiveType::Int))
                        .with_initial_default(Literal::int(7))
                        .into(),
                ])
                .build()
                .unwrap(),
        );
        let data = write_parquet(&dir, "data.parquet", vec![(
            field("id", DataType::Int64, 1),
            Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef,
        )]);
        let b_is_7 = write_parquet(&dir, "b7.parquet", vec![(
            field("b", DataType::Int32, 3),
            Arc::new(Int32Array::from(vec![7])) as ArrayRef,
        )]);
        let b_is_8 = write_parquet(&dir, "b8.parquet", vec![(
            field("b", DataType::Int32, 3),
            Arc::new(Int32Array::from(vec![8])) as ArrayRef,
        )]);
        let id_2_b_7 = write_parquet(&dir, "id2b7.parquet", vec![
            (
                field("id", DataType::Int64, 1),
                Arc::new(Int64Array::from(vec![2])) as ArrayRef,
            ),
            (
                field("b", DataType::Int32, 3),
                Arc::new(Int32Array::from(vec![7])) as ArrayRef,
            ),
        ]);

        let read = |deletes| read_ids(schema.clone(), &data, deletes, None);
        assert!(read(vec![eq_delete(&b_is_7, vec![3])]).await.is_empty());
        assert_eq!(read(vec![eq_delete(&b_is_8, vec![3])]).await, vec![1, 2, 3]);
        assert_eq!(read(vec![eq_delete(&id_2_b_7, vec![1, 3])]).await, vec![
            1, 3
        ]);
    }

    // A key nested in a struct is read through the struct, and a null parent struct makes the
    // key null, so it matches a null delete value.
    #[tokio::test]
    async fn test_eq_delete_on_struct_nested_column() {
        let dir = TempDir::new().unwrap();
        let schema = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                    NestedField::optional(
                        5,
                        "s",
                        Type::Struct(StructType::new(vec![
                            NestedField::optional(6, "sa", Type::Primitive(PrimitiveType::Int))
                                .into(),
                        ])),
                    )
                    .into(),
                ])
                .build()
                .unwrap(),
        );
        let struct_fields = Fields::from(vec![field("sa", DataType::Int32, 6)]);
        let struct_column = |values: Vec<Option<i32>>, nulls: Option<Vec<bool>>| {
            Arc::new(StructArray::new(
                struct_fields.clone(),
                vec![Arc::new(Int32Array::from(values)) as ArrayRef],
                nulls.map(Into::into),
            )) as ArrayRef
        };

        let data = write_parquet(&dir, "data.parquet", vec![
            (
                field("id", DataType::Int64, 1),
                Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef,
            ),
            (
                field("s", DataType::Struct(struct_fields.clone()), 5),
                struct_column(
                    vec![Some(1), Some(2), Some(3)],
                    Some(vec![true, false, true]),
                ),
            ),
        ]);
        let deletes = write_parquet(&dir, "deletes.parquet", vec![(
            field("s", DataType::Struct(struct_fields.clone()), 5),
            struct_column(vec![Some(1), None], None),
        )]);

        let ids = read_ids(schema, &data, vec![eq_delete(&deletes, vec![6])], None).await;
        assert_eq!(ids, vec![3]);
    }

    // A nested key the data file does not store reads as its `initial-default`, but stays null
    // where the enclosing struct is null; if the struct itself is not stored and has no
    // default, the key is null in every row.
    #[tokio::test]
    async fn test_eq_delete_on_missing_nested_column_keeps_null_structs() {
        let dir = TempDir::new().unwrap();
        let schema = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                    NestedField::optional(
                        5,
                        "s",
                        Type::Struct(StructType::new(vec![
                            NestedField::optional(6, "sa", Type::Primitive(PrimitiveType::Int))
                                .into(),
                            NestedField::optional(8, "key", Type::Primitive(PrimitiveType::Int))
                                .with_initial_default(Literal::int(7))
                                .into(),
                        ])),
                    )
                    .into(),
                ])
                .build()
                .unwrap(),
        );
        let struct_fields = Fields::from(vec![field("sa", DataType::Int32, 6)]);
        let with_struct = write_parquet(&dir, "with-struct.parquet", vec![
            (
                field("id", DataType::Int64, 1),
                Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef,
            ),
            (
                field("s", DataType::Struct(struct_fields.clone()), 5),
                Arc::new(StructArray::new(
                    struct_fields,
                    vec![Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef],
                    Some(vec![true, false, true].into()),
                )) as ArrayRef,
            ),
        ]);
        let without_struct = write_parquet(&dir, "without-struct.parquet", vec![(
            field("id", DataType::Int64, 1),
            Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef,
        )]);
        let key_deletes = |name: &str, value: Option<i32>| {
            let key_fields = Fields::from(vec![field("key", DataType::Int32, 8)]);
            write_parquet(&dir, name, vec![(
                field("s", DataType::Struct(key_fields.clone()), 5),
                Arc::new(StructArray::new(
                    key_fields,
                    vec![Arc::new(Int32Array::from(vec![value])) as ArrayRef],
                    None,
                )) as ArrayRef,
            )])
        };
        let key_is_7 = key_deletes("key7.parquet", Some(7));
        let key_is_null = key_deletes("key-null.parquet", None);

        for (data, deletes, expected) in [
            (&with_struct, &key_is_7, vec![2]),
            (&with_struct, &key_is_null, vec![1, 3]),
            (&without_struct, &key_is_7, vec![1, 2, 3]),
            (&without_struct, &key_is_null, vec![]),
        ] {
            let ids = read_ids(
                schema.clone(),
                data,
                vec![eq_delete(deletes, vec![8])],
                None,
            )
            .await;
            assert_eq!(ids, expected, "{data} with {deletes}");
        }
    }

    // The file stores the struct enclosing a missing nested key even when the only child it
    // stores was since dropped from the table schema: that struct's nulls still decide which
    // rows read the key's default.
    #[tokio::test]
    async fn test_eq_delete_on_missing_nested_column_with_dropped_sibling() {
        let dir = TempDir::new().unwrap();
        let schema = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                    NestedField::optional(
                        5,
                        "s",
                        Type::Struct(StructType::new(vec![
                            NestedField::optional(8, "key", Type::Primitive(PrimitiveType::Int))
                                .with_initial_default(Literal::int(7))
                                .into(),
                        ])),
                    )
                    .into(),
                ])
                .build()
                .unwrap(),
        );
        // `old_field` (id 9) was dropped from `s` after this file was written.
        let struct_fields = Fields::from(vec![field("old_field", DataType::Int32, 9)]);
        let data = write_parquet(&dir, "data.parquet", vec![
            (
                field("id", DataType::Int64, 1),
                Arc::new(Int64Array::from(vec![1, 2])) as ArrayRef,
            ),
            (
                field("s", DataType::Struct(struct_fields.clone()), 5),
                Arc::new(StructArray::new(
                    struct_fields,
                    vec![Arc::new(Int32Array::from(vec![10, 20])) as ArrayRef],
                    Some(vec![false, true].into()),
                )) as ArrayRef,
            ),
        ]);
        let key_fields = Fields::from(vec![field("key", DataType::Int32, 8)]);
        let deletes = write_parquet(&dir, "deletes.parquet", vec![(
            field("s", DataType::Struct(key_fields.clone()), 5),
            Arc::new(StructArray::new(
                key_fields,
                vec![Arc::new(Int32Array::from(vec![7])) as ArrayRef],
                None,
            )) as ArrayRef,
        )]);

        let ids = read_ids(schema, &data, vec![eq_delete(&deletes, vec![8])], None).await;
        assert_eq!(ids, vec![1]);
    }

    // `column_by_path` resolves a key through its parent structs; a null parent makes the
    // key null.
    #[test]
    fn test_column_by_path_propagates_struct_nulls() {
        let struct_fields = Fields::from(vec![Field::new("sa", DataType::Int32, true)]);
        let struct_array = StructArray::new(
            struct_fields.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef],
            Some(vec![true, false, true].into()),
        );
        let batch = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![Field::new(
                "s",
                DataType::Struct(struct_fields),
                true,
            )])),
            vec![Arc::new(struct_array) as ArrayRef],
        )
        .unwrap();

        let column = super::column_by_path(&batch, &["s".to_string(), "sa".to_string()]).unwrap();
        assert_eq!(
            column.as_primitive::<arrow_array::types::Int32Type>(),
            &Int32Array::from(vec![Some(1), None, Some(3)])
        );

        assert!(super::column_by_path(&batch, &["s".to_string(), "missing".to_string()]).is_err());
    }
}
