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
//! It uses the `gcloud-auth` crate which handles token caching and refresh internally.

use google_cloud_auth::credentials::CredentialsFile;
use google_cloud_auth::project::Config;
use google_cloud_auth::token::DefaultTokenSourceProvider;
use iceberg::{Error, ErrorKind, Result};
use token_source::TokenSourceProvider as _;

/// Google ADC authentication manager.
#[derive(Clone)]
pub struct GoogleAuthManager {
    credentials_json: Option<String>,
    scopes: Vec<String>,
}

impl GoogleAuthManager {
    /// Creates a new `GoogleAuthManager`.
    pub fn new(credentials_json: Option<String>, scopes: Option<Vec<String>>) -> Self {
        Self {
            credentials_json,
            scopes: scopes.unwrap_or_else(|| {
                vec!["https://www.googleapis.com/auth/cloud-platform".to_string()]
            }),
        }
    }

    /// Get token using Google ADC.
    pub async fn get_token(&self) -> Result<String> {
        let scopes: Vec<&str> = self.scopes.iter().map(|s| s.as_str()).collect();
        let config = Config::default().with_scopes(&scopes);

        let provider = if let Some(ref json) = self.credentials_json {
            let creds = CredentialsFile::new_from_str(json).await.map_err(|e| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("Failed to load Google credentials: {e}"),
                )
            })?;
            DefaultTokenSourceProvider::new_with_credentials(config, Box::new(creds))
                .await
                .map_err(|e| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!("Failed to initialize token source from credentials: {e}"),
                    )
                })?
        } else {
            DefaultTokenSourceProvider::new(config).await.map_err(|e| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Google ADC not configured. Set GOOGLE_APPLICATION_CREDENTIALS \
                         environment variable or configure a service account. Error: {e}"
                    ),
                )
            })?
        };

        // token_source.token() returns "Bearer <access_token>"
        // Strip the prefix since authenticate() in client.rs adds "Bearer " again.
        let token = provider
            .token_source()
            .token()
            .await
            .map_err(|e| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("Failed to obtain Google access token: {e}"),
                )
            })?;
        let access_token = token.strip_prefix("Bearer ").unwrap_or(&token).to_string();
        Ok(access_token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_google_auth_manager_new() {
        let auth = GoogleAuthManager::new(None, None);
        assert!(auth.credentials_json.is_none());
        assert_eq!(auth.scopes.len(), 1);
    }

    #[tokio::test]
    async fn test_invalid_credentials_json_content() {
        let auth = GoogleAuthManager::new(Some("not valid json at all".to_string()), None);
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
