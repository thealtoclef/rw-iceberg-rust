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

//! Hash-set representation of the keys of one or more equality delete files.
//!
//! Each delete row is encoded with the Arrow row format into a byte key, so applying the
//! deletes to a data batch costs one hash probe per row instead of evaluating a predicate
//! per delete key. The row format encodes a null as a distinct value, which gives exactly
//! the spec's equality-delete semantics: a null delete value matches only a null data value,
//! and a non-null delete value never matches a null data value.

use std::fmt;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, BooleanArray, make_array};
use arrow_buffer::NullBuffer;
use arrow_cast::cast::cast;
use arrow_row::{RowConverter, SortField};
use arrow_schema::{ArrowError, DataType};
use fnv::FnvHashSet;

use crate::arrow::type_to_arrow_type;
use crate::spec::{PrimitiveType, Type};
use crate::{Error, ErrorKind, Result};

/// One equality column of an [`EqDeleteSet`] key.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct EqDeleteKeyField {
    pub(crate) field_id: i32,
    pub(crate) name: String,
    /// Iceberg type of the column in the schema the delete file was evolved to.
    pub(crate) field_type: PrimitiveType,
    /// Arrow type the key column is encoded as. Data columns are cast to it before probing.
    pub(crate) data_type: DataType,
}

/// Delete keys of equality delete files that share the same equality columns.
pub(crate) struct EqDeleteSet {
    /// Key columns ordered by field id.
    fields: Vec<EqDeleteKeyField>,
    converter: RowConverter,
    keys: FnvHashSet<Box<[u8]>>,
}

impl fmt::Debug for EqDeleteSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EqDeleteSet")
            .field("fields", &self.fields)
            .field("len", &self.keys.len())
            .finish()
    }
}

impl EqDeleteSet {
    /// Creates an empty set keyed by `fields`, given as `(field_id, name, type)`.
    pub(crate) fn try_new(mut fields: Vec<(i32, String, PrimitiveType)>) -> Result<Self> {
        fields.sort_by_key(|(field_id, ..)| *field_id);
        if fields.windows(2).any(|w| w[0].0 == w[1].0) {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "Equality delete key contains a duplicated field id",
            ));
        }

        let fields = fields
            .into_iter()
            .map(|(field_id, name, field_type)| {
                let data_type = type_to_arrow_type(&Type::Primitive(field_type.clone()))?;
                Ok(EqDeleteKeyField {
                    field_id,
                    name,
                    field_type,
                    data_type,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let converter = RowConverter::new(
            fields
                .iter()
                .map(|field| SortField::new(field.data_type.clone()))
                .collect(),
        )?;

        Ok(Self {
            fields,
            converter,
            keys: FnvHashSet::default(),
        })
    }

    /// A set without key columns, used for delete files that contain no rows.
    pub(crate) fn empty() -> Self {
        Self {
            fields: Vec::new(),
            converter: RowConverter::new(Vec::new()).expect("empty row converter"),
            keys: FnvHashSet::default(),
        }
    }

    pub(crate) fn fields(&self) -> &[EqDeleteKeyField] {
        &self.fields
    }

    pub(crate) fn len(&self) -> usize {
        self.keys.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Two sets can be merged, and probed with the same encoded rows, only if their key
    /// columns have the same field ids and encoded types.
    pub(crate) fn has_same_layout(&self, other: &EqDeleteSet) -> bool {
        self.fields.len() == other.fields.len()
            && self
                .fields
                .iter()
                .zip(&other.fields)
                .all(|(l, r)| l.field_id == r.field_id && l.data_type == r.data_type)
    }

    /// Adds one delete key per row of `columns`, which must be ordered like [`Self::fields`].
    pub(crate) fn insert_columns(&mut self, columns: &[ArrayRef]) -> Result<()> {
        let rows = self.encode(columns)?;
        self.keys.reserve(rows.num_rows());
        for row in rows.iter() {
            self.keys.insert(row.as_ref().into());
        }
        Ok(())
    }

    /// Returns a mask that is `true` for every row of `columns` whose key is NOT in the set,
    /// i.e. the rows to keep. `columns` must be ordered like [`Self::fields`].
    pub(crate) fn keep_mask(
        &self,
        columns: &[ArrayRef],
    ) -> std::result::Result<BooleanArray, ArrowError> {
        let rows = self.encode(columns)?;
        Ok(rows
            .iter()
            .map(|row| Some(!self.keys.contains(row.as_ref())))
            .collect())
    }

    /// Merges sets that share the same layout. A single set is returned as is.
    pub(crate) fn union(sets: &[Arc<EqDeleteSet>]) -> Result<Arc<EqDeleteSet>> {
        let Some((first, rest)) = sets.split_first() else {
            return Ok(Arc::new(Self::empty()));
        };
        if rest.is_empty() {
            return Ok(first.clone());
        }

        let mut merged = Self::try_new(
            first
                .fields
                .iter()
                .map(|f| (f.field_id, f.name.clone(), f.field_type.clone()))
                .collect(),
        )?;
        merged
            .keys
            .reserve(sets.iter().map(|set| set.len()).sum::<usize>());
        for set in sets {
            if !first.has_same_layout(set) {
                return Err(Error::new(
                    ErrorKind::Unexpected,
                    "Cannot merge equality delete sets with different key columns",
                ));
            }
            merged.keys.extend(set.keys.iter().cloned());
        }
        Ok(Arc::new(merged))
    }

    fn encode(&self, columns: &[ArrayRef]) -> std::result::Result<arrow_row::Rows, ArrowError> {
        if columns.len() != self.fields.len() {
            return Err(ArrowError::InvalidArgumentError(format!(
                "Equality delete key has {} columns but {} were given",
                self.fields.len(),
                columns.len()
            )));
        }

        let columns = columns
            .iter()
            .zip(&self.fields)
            .map(|(column, field)| {
                if column.data_type() == &field.data_type {
                    Ok(column.clone())
                } else {
                    cast(column, &field.data_type)
                }
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;

        self.converter.convert_columns(&columns)
    }
}

/// Adds `nulls` (typically the nulls of an enclosing struct) to the nulls of `array`. A
/// field of a null struct reads as null, whatever value is stored for it.
pub(crate) fn with_nulls(
    array: ArrayRef,
    nulls: Option<NullBuffer>,
) -> std::result::Result<ArrayRef, ArrowError> {
    match NullBuffer::union(nulls.as_ref(), array.nulls()) {
        Some(nulls) if Some(&nulls) != array.nulls() => Ok(make_array(
            array.to_data().into_builder().nulls(Some(nulls)).build()?,
        )),
        _ => Ok(array),
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{Float64Array, Int32Array, Int64Array, StringArray};

    use super::*;

    fn long_set(values: Vec<Option<i64>>) -> EqDeleteSet {
        let mut set =
            EqDeleteSet::try_new(vec![(1, "id".to_string(), PrimitiveType::Long)]).unwrap();
        set.insert_columns(&[Arc::new(Int64Array::from(values)) as ArrayRef])
            .unwrap();
        set
    }

    #[test]
    fn test_keep_mask_single_column() {
        let set = long_set(vec![Some(2), Some(4), Some(4)]);
        assert_eq!(set.len(), 2);

        let data = Arc::new(Int64Array::from(vec![1, 2, 3, 4])) as ArrayRef;
        let keep = set.keep_mask(&[data]).unwrap();
        assert_eq!(keep, BooleanArray::from(vec![true, false, true, false]));
    }

    // A null data value never equals a non-null delete value; a null delete value matches
    // only null data values.
    #[test]
    fn test_keep_mask_null_semantics() {
        let set = long_set(vec![Some(2)]);
        let data = Arc::new(Int64Array::from(vec![None, Some(2)])) as ArrayRef;
        assert_eq!(
            set.keep_mask(std::slice::from_ref(&data)).unwrap(),
            BooleanArray::from(vec![true, false])
        );

        let set = long_set(vec![None]);
        assert_eq!(
            set.keep_mask(&[data]).unwrap(),
            BooleanArray::from(vec![false, true])
        );
    }

    #[test]
    fn test_keep_mask_composite_key() {
        // Fields are given out of order on purpose: keys are ordered by field id.
        let mut set = EqDeleteSet::try_new(vec![
            (3, "status".to_string(), PrimitiveType::String),
            (1, "id".to_string(), PrimitiveType::Long),
        ])
        .unwrap();
        assert_eq!(
            set.fields().iter().map(|f| f.field_id).collect::<Vec<_>>(),
            vec![1, 3]
        );
        set.insert_columns(&[
            Arc::new(Int64Array::from(vec![1, 2])) as ArrayRef,
            Arc::new(StringArray::from(vec![Some("X"), None])) as ArrayRef,
        ])
        .unwrap();

        let keep = set
            .keep_mask(&[
                Arc::new(Int64Array::from(vec![1, 1, 2, 2])) as ArrayRef,
                Arc::new(StringArray::from(vec![
                    Some("X"),
                    Some("Y"),
                    None,
                    Some("X"),
                ])) as ArrayRef,
            ])
            .unwrap();
        assert_eq!(keep, BooleanArray::from(vec![false, true, false, true]));
    }

    // Data files written before an int -> long promotion still store int32.
    #[test]
    fn test_keep_mask_casts_promoted_data_column() {
        let set = long_set(vec![Some(7)]);
        let data = Arc::new(Int32Array::from(vec![6, 7])) as ArrayRef;
        assert_eq!(
            set.keep_mask(&[data]).unwrap(),
            BooleanArray::from(vec![true, false])
        );
    }

    // Floats compare by their total-order bit pattern, like the Arrow `neq` kernel the
    // predicate-based implementation used: -0.0 and 0.0 are distinct, NaN matches NaN.
    #[test]
    fn test_keep_mask_float_semantics() {
        let mut set =
            EqDeleteSet::try_new(vec![(1, "f".to_string(), PrimitiveType::Double)]).unwrap();
        set.insert_columns(&[Arc::new(Float64Array::from(vec![0.0, f64::NAN])) as ArrayRef])
            .unwrap();

        let data = Arc::new(Float64Array::from(vec![0.0, -0.0, f64::NAN, 1.0])) as ArrayRef;
        assert_eq!(
            set.keep_mask(&[data]).unwrap(),
            BooleanArray::from(vec![false, true, false, true])
        );
    }

    #[test]
    fn test_union() {
        let a = Arc::new(long_set(vec![Some(1), Some(2)]));
        let b = Arc::new(long_set(vec![Some(2), Some(3)]));

        assert!(Arc::ptr_eq(
            &EqDeleteSet::union(std::slice::from_ref(&a)).unwrap(),
            &a
        ));

        let merged = EqDeleteSet::union(&[a, b]).unwrap();
        assert_eq!(merged.len(), 3);
        let data = Arc::new(Int64Array::from(vec![1, 2, 3, 4])) as ArrayRef;
        assert_eq!(
            merged.keep_mask(&[data]).unwrap(),
            BooleanArray::from(vec![false, false, false, true])
        );
    }

    #[test]
    fn test_union_rejects_different_layouts() {
        let a = Arc::new(long_set(vec![Some(1)]));
        let mut b =
            EqDeleteSet::try_new(vec![(2, "other".to_string(), PrimitiveType::Long)]).unwrap();
        b.insert_columns(&[Arc::new(Int64Array::from(vec![1])) as ArrayRef])
            .unwrap();

        assert!(EqDeleteSet::union(&[a, Arc::new(b)]).is_err());
    }
}
