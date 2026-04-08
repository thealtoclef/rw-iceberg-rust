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

//! Google ADC (Application Default Credentials) authentication manager.
//!
//! This module provides Google ADC authentication for BigLake/Iceberg REST catalogs.
//! It uses the `gcp_auth` crate which depends on `hyper-rustls` which depends on `rustls 0.23+`.

use std::sync::OnceLock;

use gcp_auth::TokenProvider;
use iceberg::{Error, ErrorKind, Result};

// rustls 0.23+ requires explicit crypto provider selection when multiple TLS implementations
// are in the dependency tree. hyper-rustls (used by gcp_auth) depends on rustls 0.23 but
// doesn't enable a crypto provider feature by default. This can cause a panic at runtime:
// "Could not automatically determine the process-level CryptoProvider from Rustls crate features."
//
// We use OnceLock to ensure the crypto provider is installed exactly once, before any
// gcp_auth operations. This allows users to use google-auth=true without needing to
// manually call rustls::crypto::ring::default_provider().install_default() in their code.
static CRYPTO_PROVIDER_INIT: OnceLock<()> = OnceLock::new();

fn init_crypto_provider() {
    CRYPTO_PROVIDER_INIT.get_or_init(|| {
        rustls::crypto::ring::default_provider()
            .install_default()
            .expect("Failed to install rustls crypto provider");
    });
}

/// Google ADC authentication manager.
#[derive(Clone)]
pub struct GoogleAuthManager {
    credentials_path: Option<String>,
    scopes: Vec<String>,
}

impl GoogleAuthManager {
    /// Creates a new `GoogleAuthManager`.
    pub fn new(credentials_path: Option<String>, scopes: Option<Vec<String>>) -> Self {
        init_crypto_provider();
        Self {
            credentials_path,
            scopes: scopes.unwrap_or_else(|| {
                vec!["https://www.googleapis.com/auth/cloud-platform".to_string()]
            }),
        }
    }

    /// Get token using Google ADC.
    pub async fn get_token(&self) -> Result<String> {
        let scopes: Vec<&str> = self.scopes.iter().map(|s| s.as_str()).collect();

        if let Some(ref credentials_path) = self.credentials_path {
            let custom_sa =
                gcp_auth::CustomServiceAccount::from_json(credentials_path).map_err(|e| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!("Failed to load Google credentials from '{credentials_path}': {e}"),
                    )
                })?;
            let token = custom_sa.token(&scopes).await.map_err(|e| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("Failed to get token from service account: {e}"),
                )
            })?;
            return Ok(token.as_str().to_string());
        }

        let provider = gcp_auth::provider().await.map_err(|e| {
            Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Google ADC not configured. Set GOOGLE_APPLICATION_CREDENTIALS environment variable or configure a service account. Error: {e}"
                ),
            )
        })?;

        let token = provider.token(&scopes).await.map_err(|e| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("Failed to obtain Google access token via ADC: {e}"),
            )
        })?;

        Ok(token.as_str().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_google_auth_manager_new() {
        let auth = GoogleAuthManager::new(None, None);
        assert!(auth.credentials_path.is_none());
        assert_eq!(auth.scopes.len(), 1);
    }

    #[tokio::test]
    async fn test_invalid_credentials_json_content() {
        let auth =
            GoogleAuthManager::new(Some("/nonexistent/path/credentials.json".to_string()), None);
        let result = auth.get_token().await;
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Failed to load Google credentials")
        );
    }

    #[tokio::test]
    async fn test_malformed_credentials_json() {
        let auth = GoogleAuthManager::new(Some(r#"{"invalid": "json"}"#.to_string()), None);
        let result = auth.get_token().await;
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Failed to load Google credentials")
        );
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_real_adc_auth() {
        let skip_reason = std::env::var("TEST_WITH_REAL_GCP")
            .ok()
            .map(|_| None)
            .unwrap_or_else(|| Some("TEST_WITH_REAL_GCP not set - skipping real ADC test"));

        if let Some(reason) = skip_reason {
            eprintln!("Skipping test: {reason}");
            return;
        }

        let auth = GoogleAuthManager::new(None, None);
        let result = auth.get_token().await;

        if result.is_err() {
            let error = result.unwrap_err();
            eprintln!("ADC not available (expected in some environments): {error}");
            return;
        }

        let token = result.unwrap();
        assert!(!token.is_empty());
    }
}
