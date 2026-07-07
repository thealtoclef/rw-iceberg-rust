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

//! Run [`RemoveDanglingDeleteFilesAction`] against a BigLake REST catalog,
//! authenticating to both the catalog and GCS with a single ADC access token.
//!
//! Run with:
//!   cargo run -p iceberg-examples --features storage-gcs \
//!       --example remove-dangling-deletes

use std::collections::HashMap;
use std::process::Command;
use std::sync::Arc;

use iceberg::actions::RemoveDanglingDeleteFilesAction;
use iceberg::{Catalog, CatalogBuilder, TableIdent};
use iceberg_catalog_rest::{REST_CATALOG_PROP_URI, REST_CATALOG_PROP_WAREHOUSE, RestCatalogBuilder};

const CATALOG_URI: &str = "https://biglake.googleapis.com/iceberg/v1/restcatalog";
const PROJECT_ID: &str = "cake-data-non-production";
const WAREHOUSE: &str = "bq://projects/cake-data-non-production/locations/asia-southeast1";
const NAMESPACE: &str = "cake_repro";
const TABLE_NAME: &str = "tb_too_many_equality_deletes";

/// Fetches a GCP OAuth2 access token from local Application Default Credentials
/// by shelling out to the `gcloud` CLI. Swap this out for the `gcp_auth` crate
/// (or a service-account flow) in a non-interactive/production deployment.
fn adc_access_token() -> String {
    let output = Command::new("gcloud")
        .args(["auth", "application-default", "print-access-token"])
        .output()
        .expect("failed to run `gcloud`; is the gcloud CLI installed and on PATH?");

    if !output.status.success() {
        panic!(
            "`gcloud auth application-default print-access-token` failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    String::from_utf8(output.stdout)
        .expect("token is not valid UTF-8")
        .trim()
        .to_string()
}

#[tokio::main]
async fn main() {
    // 1. Obtain one access token from local ADC and reuse it for both services.
    let token = adc_access_token();

    // 2. Build the props consumed by the REST catalog and the GCS FileIO.
    //    - `token` / `header.*`        -> authenticate to the BigLake REST catalog
    //    - `gcs.oauth2.token` / `gcs.*` -> authenticate the GCS object store reads/writes
    let props = HashMap::from([
        (REST_CATALOG_PROP_URI.to_string(), CATALOG_URI.to_string()),
        (REST_CATALOG_PROP_WAREHOUSE.to_string(), WAREHOUSE.to_string()),
        // REST catalog: Bearer auth + the project header BigLake requires.
        ("token".to_string(), token.clone()),
        (
            "header.x-goog-user-project".to_string(),
            PROJECT_ID.to_string(),
        ),
        // GCS FileIO: use the same token directly, and pin the project so we
        // never fall back to GoogleAuthManager / VM metadata / config discovery.
        ("gcs.oauth2.token".to_string(), token),
        ("gcs.project-id".to_string(), PROJECT_ID.to_string()),
        ("gcs.user-project".to_string(), PROJECT_ID.to_string()),
        ("gcs.disable-vm-metadata".to_string(), "true".to_string()),
        ("gcs.disable-config-load".to_string(), "true".to_string()),
    ]);

    let catalog: Arc<dyn Catalog> = Arc::new(
        RestCatalogBuilder::default()
            .load("biglake", props)
            .await
            .expect("failed to initialize BigLake REST catalog"),
    );

    // 3. Run the maintenance action against the target table.
    let table_ident = TableIdent::from_strs([NAMESPACE, TABLE_NAME]).unwrap();

    println!("Scanning {NAMESPACE}.{TABLE_NAME} for dangling delete files...");
    let removed = RemoveDanglingDeleteFilesAction::new(catalog, table_ident)
        .to_branch("main") // optional; "main" is the default
        .execute()
        .await
        .expect("RemoveDanglingDeleteFilesAction failed");

    println!("Removed {removed} dangling delete file(s).");
}
