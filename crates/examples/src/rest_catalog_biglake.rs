use std::collections::HashMap;

use iceberg::spec::{NestedField, PrimitiveType, Schema, Type};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
use iceberg_catalog_rest::{
    GOOGLE_AUTH_PROP, REST_CATALOG_PROP_URI, REST_CATALOG_PROP_WAREHOUSE, RestCatalogBuilder,
};

static GCP_PROJECT_ID: &str = "rainbow-data-production-483609";
static BIGLAKE_URI: &str = "https://biglake.googleapis.com/iceberg/v1/restcatalog";
static WAREHOUSE: &str = "bq://projects/rainbow-data-production-483609";
static GCS_BUCKET_LOCATION: &str = "gs://rainbow-data-production-iceberg/test_iceberg_compactor_v2";
static BQ_CONNECTION: &str =
    "projects/rainbow-data-production-483609/locations/us/connections/iceberg_conn";
static USER_PROJECT_HEADER: &str = "x-goog-user-project";

static NAMESPACE_NAME: &str = "test_iceberg_compactor_v2";
static TABLE_NAME: &str = "compaction_demo";

#[tokio::main]
async fn main() {
    println!("BigLake Metastore Iceberg REST Catalog Example");
    println!("============================================");
    println!();

    let mut props = HashMap::new();
    props.insert(REST_CATALOG_PROP_URI.to_string(), BIGLAKE_URI.to_string());
    props.insert(
        REST_CATALOG_PROP_WAREHOUSE.to_string(),
        WAREHOUSE.to_string(),
    );
    props.insert(GOOGLE_AUTH_PROP.to_string(), "true".to_string());
    props.insert(
        format!("header.{USER_PROJECT_HEADER}"),
        GCP_PROJECT_ID.to_string(),
    );

    println!("Configuration:");
    println!("  URI: {BIGLAKE_URI}");
    println!("  Warehouse: {WAREHOUSE}");
    println!("  Auth: google-auth=true (ADC)");
    println!("  Header: {USER_PROJECT_HEADER}={GCP_PROJECT_ID}");
    println!();

    println!("Connecting to BigLake...");
    let catalog = RestCatalogBuilder::default()
        .load("biglake", props)
        .await
        .unwrap();

    println!("✓ Connected successfully!");
    println!();

    let namespace_ident = NamespaceIdent::from_vec(vec![NAMESPACE_NAME.to_string()]).unwrap();

    if catalog.namespace_exists(&namespace_ident).await.unwrap() {
        println!("Namespace '{NAMESPACE_NAME}' already exists, checking for existing table...");
        let table_ident = TableIdent::new(namespace_ident.clone(), TABLE_NAME.to_string());

        if catalog.table_exists(&table_ident).await.unwrap() {
            println!("Table '{TABLE_NAME}' already exists, dropping...");
            catalog.drop_table(&table_ident).await.unwrap();
            println!("  ✓ Dropped");
        }
    } else {
        println!("Creating namespace '{NAMESPACE_NAME}'...");
        let created_ns = catalog
            .create_namespace(
                &namespace_ident,
                HashMap::from([("location".to_owned(), GCS_BUCKET_LOCATION.to_owned())]),
            )
            .await
            .unwrap();
        println!("✓ Namespace created: {created_ns:?}");
    }
    println!();

    println!("Listing all namespaces...");
    let namespaces = catalog.list_namespaces(None).await.unwrap();
    println!("  Found {} namespace(s)", namespaces.len());
    for ns in &namespaces {
        println!("  - {ns:?}");
    }
    println!();

    let table_ident = TableIdent::new(namespace_ident.clone(), TABLE_NAME.to_string());

    if catalog.table_exists(&table_ident).await.unwrap() {
        println!("Table '{TABLE_NAME}' already exists, dropping...");
        catalog.drop_table(&table_ident).await.unwrap();
        println!("  ✓ Dropped");
    }

    let table_schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::optional(3, "value", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .with_schema_id(0)
        .build()
        .unwrap();

    let table_creation = TableCreation::builder()
        .name(table_ident.name.clone())
        .schema(table_schema.clone())
        .properties(HashMap::from([(
            "bq_connection".to_string(),
            BQ_CONNECTION.to_string(),
        )]))
        .build();

    println!("Creating table '{NAMESPACE_NAME}.{TABLE_NAME}'...");
    println!("  Table properties: bq_connection={BQ_CONNECTION}");
    let created_table = catalog
        .create_table(&namespace_ident, table_creation)
        .await
        .unwrap();
    println!("✓ Table created: {created_table:?}");
    println!();

    println!("Listing tables in namespace '{NAMESPACE_NAME}'...");
    let tables = catalog.list_tables(&namespace_ident).await.unwrap();
    println!("  Found {} table(s)", tables.len());
    for t in &tables {
        println!("  - {t:?}");
    }
    println!();

    assert!(
        tables.contains(&table_ident),
        "Table should be in namespace"
    );

    println!("============================================");
    println!("Example completed successfully!");
    println!();
    println!("To query this table, use Spark or another Iceberg-compatible engine:");
    println!("  SELECT * FROM {NAMESPACE_NAME}.{TABLE_NAME}");
}
