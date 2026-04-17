package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// minimalJPEG is a valid 1x1 pixel JPEG image.
var minimalJPEG = []byte{
	0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
	0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
	0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
	0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
	0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
	0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
	0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
	0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01,
	0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00,
	0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
	0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01, 0x03,
	0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D,
	0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06,
	0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
	0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72,
	0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
	0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45,
	0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
	0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75,
	0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
	0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3,
	0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6,
	0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9,
	0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
	0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4,
	0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01,
	0x00, 0x00, 0x3F, 0x00, 0x7B, 0x94, 0x11, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xD9,
}

// minimalPNG is a valid 1x1 pixel white PNG image.
var minimalPNG = []byte{
	0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
	0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, // IHDR chunk
	0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, // 1x1 pixel
	0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53, // 8-bit RGB
	0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, // IDAT chunk
	0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00,
	0x00, 0x00, 0x02, 0x00, 0x01, 0xE2, 0x21, 0xBC,
	0x33, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, // IEND chunk
	0x44, 0xAE, 0x42, 0x60, 0x82,
}

// inferenceStubResponse is a realistic JSON response matching PredictionResponse.
const inferenceStubResponse = `{
	"top_5_predictions": [
		{"class_name": "cat", "confidence_percent": 92.50},
		{"class_name": "dog", "confidence_percent": 3.20},
		{"class_name": "tiger", "confidence_percent": 1.80},
		{"class_name": "lion", "confidence_percent": 1.10},
		{"class_name": "leopard", "confidence_percent": 0.65}
	],
	"inference_time_ms": 3.14
}`

// newInferenceStub creates an httptest server that mimics the inference service.
func newInferenceStub(t *testing.T, statusCode int) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/internal/predict" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(statusCode)
		if statusCode == http.StatusOK {
			fmt.Fprint(w, inferenceStubResponse)
		} else {
			fmt.Fprint(w, `{"detail":"internal error"}`)
		}
	}))
}

// buildMultipartRequest creates a multipart form body with the given file content.
func buildMultipartRequest(t *testing.T, fieldName, fileName string, content []byte) (*bytes.Buffer, string) {
	t.Helper()
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)

	if fieldName != "" {
		part, err := writer.CreateFormFile(fieldName, fileName)
		if err != nil {
			t.Fatalf("failed to create form file: %v", err)
		}
		if _, err := io.Copy(part, bytes.NewReader(content)); err != nil {
			t.Fatalf("failed to write file content: %v", err)
		}
	}

	if err := writer.Close(); err != nil {
		t.Fatalf("failed to close writer: %v", err)
	}
	return &body, writer.FormDataContentType()
}

// testConfig returns a config suitable for testing.
func testConfig(inferenceURL string) config {
	return config{
		InferenceURL:     inferenceURL,
		MaxUploadBytes:   1 * 1024 * 1024, // 1 MB for tests
		MaxUploadMB:      1,
		Port:             "0",
		InferenceTimeout: 5 * time.Second,
	}
}

// setupApp creates a Fiber app configured for testing.
func setupApp(cfg config) *fiber.App {
	app := fiber.New(fiber.Config{
		BodyLimit:    cfg.MaxUploadBytes + 4096, // a bit extra for multipart overhead
		ErrorHandler: customErrorHandler,
	})
	app.Get("/healthz", healthHandler)
	app.Post("/api/v1/classify", classifyHandler(cfg))
	return app
}

func TestClassifyEndpoint(t *testing.T) {
	tests := []struct {
		name           string
		setupStub      func(t *testing.T) *httptest.Server
		buildRequest   func(t *testing.T) (*bytes.Buffer, string)
		expectedStatus int
		expectedError  string
	}{
		{
			name: "happy path JPEG",
			setupStub: func(t *testing.T) *httptest.Server {
				return newInferenceStub(t, http.StatusOK)
			},
			buildRequest: func(t *testing.T) (*bytes.Buffer, string) {
				return buildMultipartRequest(t, "file", "test.jpg", minimalJPEG)
			},
			expectedStatus: 200,
		},
		{
			name: "happy path PNG",
			setupStub: func(t *testing.T) *httptest.Server {
				return newInferenceStub(t, http.StatusOK)
			},
			buildRequest: func(t *testing.T) (*bytes.Buffer, string) {
				return buildMultipartRequest(t, "file", "test.png", minimalPNG)
			},
			expectedStatus: 200,
		},
		{
			name: "bad MIME type text file",
			setupStub: func(t *testing.T) *httptest.Server {
				return newInferenceStub(t, http.StatusOK)
			},
			buildRequest: func(t *testing.T) (*bytes.Buffer, string) {
				textContent := []byte("This is a plain text file, not an image.")
				return buildMultipartRequest(t, "file", "readme.txt", textContent)
			},
			expectedStatus: 415,
			expectedError:  "unsupported media type",
		},
		{
			name: "oversized file",
			setupStub: func(t *testing.T) *httptest.Server {
				return newInferenceStub(t, http.StatusOK)
			},
			buildRequest: func(t *testing.T) (*bytes.Buffer, string) {
				// Create a fake JPEG header followed by padding to exceed 1 MB.
				// Start with JPEG magic bytes so it passes MIME sniffing.
				bigFile := make([]byte, 1*1024*1024+1)
				copy(bigFile, minimalJPEG)
				return buildMultipartRequest(t, "file", "huge.jpg", bigFile)
			},
			expectedStatus: 413,
			expectedError:  "file too large",
		},
		{
			name: "inference service down",
			setupStub: func(t *testing.T) *httptest.Server {
				// Start and immediately close the server to simulate connection refused.
				stub := newInferenceStub(t, http.StatusOK)
				stub.Close()
				return stub
			},
			buildRequest: func(t *testing.T) (*bytes.Buffer, string) {
				return buildMultipartRequest(t, "file", "test.jpg", minimalJPEG)
			},
			expectedStatus: 502,
			expectedError:  "inference service unavailable",
		},
		{
			name: "inference returns 500",
			setupStub: func(t *testing.T) *httptest.Server {
				return newInferenceStub(t, http.StatusInternalServerError)
			},
			buildRequest: func(t *testing.T) (*bytes.Buffer, string) {
				return buildMultipartRequest(t, "file", "test.jpg", minimalJPEG)
			},
			expectedStatus: 502,
			expectedError:  "inference error",
		},
		{
			name: "no file attached",
			setupStub: func(t *testing.T) *httptest.Server {
				return newInferenceStub(t, http.StatusOK)
			},
			buildRequest: func(t *testing.T) (*bytes.Buffer, string) {
				// Send a multipart form with no file field.
				var body bytes.Buffer
				writer := multipart.NewWriter(&body)
				_ = writer.WriteField("other", "value")
				writer.Close()
				return &body, writer.FormDataContentType()
			},
			expectedStatus: 400,
			expectedError:  "missing file",
		},
		{
			name: "empty file",
			setupStub: func(t *testing.T) *httptest.Server {
				return newInferenceStub(t, http.StatusOK)
			},
			buildRequest: func(t *testing.T) (*bytes.Buffer, string) {
				return buildMultipartRequest(t, "file", "empty.jpg", []byte{})
			},
			expectedStatus: 415,
			expectedError:  "unsupported media type",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			stub := tc.setupStub(t)
			// Only defer close if the stub was not already closed.
			if tc.name != "inference service down" {
				defer stub.Close()
			}

			cfg := testConfig(stub.URL)
			app := setupApp(cfg)

			body, contentType := tc.buildRequest(t)

			req := httptest.NewRequest(http.MethodPost, "/api/v1/classify", body)
			req.Header.Set("Content-Type", contentType)

			resp, err := app.Test(req, -1)
			if err != nil {
				t.Fatalf("app.Test failed: %v", err)
			}
			defer resp.Body.Close()

			if resp.StatusCode != tc.expectedStatus {
				respBody, _ := io.ReadAll(resp.Body)
				t.Errorf("expected status %d, got %d; body: %s",
					tc.expectedStatus, resp.StatusCode, string(respBody))
				return
			}

			// Parse response as JSON.
			respBody, err := io.ReadAll(resp.Body)
			if err != nil {
				t.Fatalf("failed to read response body: %v", err)
			}

			var result map[string]interface{}
			if err := json.Unmarshal(respBody, &result); err != nil {
				t.Fatalf("response is not valid JSON: %v; body: %s", err, string(respBody))
			}

			// For error cases, verify the error field.
			if tc.expectedError != "" {
				errField, ok := result["error"].(string)
				if !ok {
					t.Errorf("expected 'error' field in response, got: %s", string(respBody))
					return
				}
				if !strings.Contains(errField, tc.expectedError) {
					t.Errorf("expected error containing %q, got %q", tc.expectedError, errField)
				}
			}

			// For happy path, verify prediction structure.
			if tc.expectedStatus == 200 {
				if _, ok := result["top_5_predictions"]; !ok {
					t.Errorf("expected 'top_5_predictions' in response, got: %s", string(respBody))
				}
				if _, ok := result["inference_time_ms"]; !ok {
					t.Errorf("expected 'inference_time_ms' in response, got: %s", string(respBody))
				}
			}
		})
	}
}

func TestHealthEndpoint(t *testing.T) {
	cfg := testConfig("http://localhost:9999")
	app := setupApp(cfg)

	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	resp, err := app.Test(req, -1)
	if err != nil {
		t.Fatalf("app.Test failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		t.Errorf("expected 200, got %d", resp.StatusCode)
	}

	body, _ := io.ReadAll(resp.Body)
	var result map[string]interface{}
	if err := json.Unmarshal(body, &result); err != nil {
		t.Fatalf("response is not valid JSON: %v", err)
	}

	if result["status"] != "ok" {
		t.Errorf("expected status 'ok', got %v", result["status"])
	}
}
