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

//! Remove dangling delete files action.
//!
//! Removes position delete file entries from manifests when their referenced
//! data file no longer exists in the current snapshot. This reduces metadata
//! overhead and storage consumption for tables with high CDC throughput.

use std::collections::HashSet;
use std::sync::Arc;

use crate::spec::{DataContentType, DataFile, MAIN_BRANCH};
use crate::transaction::ApplyTransactionAction;
use crate::transaction::Transaction;
use crate::{Catalog, Error, ErrorKind, Result, TableIdent};

/// Action to remove dangling position delete files from a table.
///
/// A position delete file becomes "dangling" when the data file it references
/// (via `referenced_data_file`) has been removed by compaction or partition
/// expiration. These orphaned delete entries still consume metadata space and
/// increase query planning cost.
///
/// # Example
///
/// ```ignore
/// let removed = RemoveDanglingDeleteFilesAction::new(catalog, table_ident)
///     .to_branch("main")
///     .execute()
///     .await?;
/// println!("Removed {} dangling delete files", removed);
/// ```
pub struct RemoveDanglingDeleteFilesAction {
    catalog: Arc<dyn Catalog>,
    table_ident: TableIdent,
    to_branch: String,
}

impl RemoveDanglingDeleteFilesAction {
    /// Creates a new action for the given catalog and table.
    pub fn new(catalog: Arc<dyn Catalog>, table_ident: TableIdent) -> Self {
        Self {
            catalog,
            table_ident,
            to_branch: MAIN_BRANCH.to_string(),
        }
    }

    /// Sets the branch to operate on.
    pub fn to_branch(mut self, branch: impl Into<String>) -> Self {
        self.to_branch = branch.into();
        self
    }

    /// Executes the action, returning the number of dangling delete files removed.
    pub async fn execute(self) -> Result<usize> {
        let table = self.catalog.load_table(&self.table_ident).await?;
        let Some(snapshot) = table.metadata().snapshot_for_ref(&self.to_branch) else {
            return Ok(0);
        };

        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await?;

        let mut data_file_paths: HashSet<String> = HashSet::new();
        let mut dangling: Vec<DataFile> = Vec::new();

        for mf in manifest_list.entries() {
            let manifest = mf.load_manifest(table.file_io()).await?;
            let (entries, _) = manifest.into_parts();

            for entry in entries {
                match entry.content_type() {
                    DataContentType::Data => {
                        data_file_paths.insert(entry.data_file().file_path().to_string());
                    }
                    DataContentType::PositionDeletes => {
                        let df = entry.data_file();
                        if df.referenced_data_file().is_some() {
                            dangling.push(df.clone());
                        }
                    }
                    _ => {}
                }
            }
        }

        dangling.retain(|df| {
            df.referenced_data_file()
                .map_or(false, |p| !data_file_paths.contains(&p))
        });

        if dangling.is_empty() {
            return Ok(0);
        }

        let dangling_count = dangling.len();

        // Build a rewrite-files transaction that removes the dangling delete
        // entries from the manifest without adding any new data.
        let txn = Transaction::new(&table);
        let branch = self.to_branch.clone();
        let action = txn
            .rewrite_files()
            .delete_files(dangling)
            .set_target_branch(branch);

        let txn = action
            .apply(txn)
            .map_err(|e| Error::new(ErrorKind::Unexpected, format!("Failed to build rewrite action: {e}")))?;

        txn.commit(self.catalog.as_ref()).await?;

        Ok(dangling_count)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use crate::catalog::memory::{MemoryCatalogBuilder, MEMORY_CATALOG_WAREHOUSE};
    use crate::catalog::{Catalog, CatalogBuilder};
    use crate::spec::{
        DataContentType, DataFileBuilder, DataFileFormat, NestedField, PrimitiveType, Schema,
        Type, MAIN_BRANCH,
    };
    use crate::table::Table;
    use crate::transaction::ApplyTransactionAction;
    use crate::transaction::Transaction;
    use crate::{NamespaceIdent, TableCreation, TableIdent};

    use super::RemoveDanglingDeleteFilesAction;

    fn simple_schema() -> Schema {
        Schema::builder()
            .with_schema_id(0)
            .with_identifier_field_ids(vec![1])
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap()
    }

    async fn create_test_table(
        catalog: &Arc<dyn Catalog>,
        table_name: &str,
    ) -> (TableIdent, Table) {
        let ns = NamespaceIdent::new("test_ns".into());
        catalog.create_namespace(&ns, HashMap::new()).await.ok();

        let table_ident = TableIdent::new(ns, table_name.to_string());
        let table_creation = TableCreation::builder()
            .name(table_ident.name().into())
            .schema(simple_schema())
            .build();

        catalog.create_table(&table_ident.namespace, table_creation).await.unwrap();

        let table = catalog.load_table(&table_ident).await.unwrap();
        (table_ident, table)
    }

    async fn commit_data_file(
        catalog: &Arc<dyn Catalog>,
        table: &Table,
        file_path: &str,
    ) -> Table {
        let data_file = DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(file_path.to_string())
            .file_format(DataFileFormat::Parquet)
            .record_count(1)
            .file_size_in_bytes(100)
            .build()
            .unwrap();

        let txn = Transaction::new(table);
        let action = txn
            .rewrite_files()
            .add_data_files(vec![data_file])
            .set_target_branch(MAIN_BRANCH.to_string());
        let txn = action.apply(txn).unwrap();
        txn.commit(catalog.as_ref()).await.unwrap()
    }

    async fn build_catalog() -> Arc<dyn Catalog> {
        let warehouse = "memory://test/";
        let catalog = MemoryCatalogBuilder::default()
            .load(
                "memory",
                HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.to_string())]),
            )
            .await
            .unwrap();
        Arc::new(catalog)
    }

    #[tokio::test]
    async fn test_remove_dangling_delete_files() {
        let catalog = build_catalog().await;
        let (table_ident, table) = create_test_table(&catalog, "test_dangling").await;
        let table = commit_data_file(&catalog, &table, "memory://test/data-1.parquet").await;

        let pos_delete = DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path("memory://test/pos-del-1.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .record_count(1)
            .file_size_in_bytes(100)
            .referenced_data_file(Some("memory://test/nonexistent.parquet".to_string()))
            .build()
            .unwrap();

        let txn = Transaction::new(&table);
        let action = txn
            .rewrite_files()
            .add_data_files(vec![pos_delete])
            .set_target_branch(MAIN_BRANCH.to_string());
        let txn = action.apply(txn).unwrap();
        txn.commit(catalog.as_ref()).await.unwrap();

        let removed = RemoveDanglingDeleteFilesAction::new(catalog.clone(), table_ident)
            .execute()
            .await
            .unwrap();
        assert_eq!(removed, 1);
    }

    #[tokio::test]
    async fn test_remove_dangling_delete_files_none_when_referenced() {
        let catalog = build_catalog().await;
        let (table_ident, table) = create_test_table(&catalog, "test_referenced").await;

        let data_file_path = "memory://test/data-1.parquet";
        let table = commit_data_file(&catalog, &table, data_file_path).await;

        let pos_delete = DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path("memory://test/pos-del-1.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .record_count(1)
            .file_size_in_bytes(100)
            .referenced_data_file(Some(data_file_path.to_string()))
            .build()
            .unwrap();

        let txn = Transaction::new(&table);
        let action = txn
            .rewrite_files()
            .add_data_files(vec![pos_delete])
            .set_target_branch(MAIN_BRANCH.to_string());
        let txn = action.apply(txn).unwrap();
        txn.commit(catalog.as_ref()).await.unwrap();

        let removed = RemoveDanglingDeleteFilesAction::new(catalog.clone(), table_ident)
            .execute()
            .await
            .unwrap();
        assert_eq!(removed, 0);
    }
}
