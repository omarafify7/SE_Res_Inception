package main

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/gofiber/fiber/v2"
)

// config holds runtime configuration parsed from environment variables.
type config struct {
	InferenceURL     string
	MaxUploadBytes   int
	MaxUploadMB      int
	Port             string
	InferenceTimeout time.Duration
}

func loadConfig() config {
	inferenceURL := os.Getenv("INFERENCE_URL")
	if inferenceURL == "" {
		inferenceURL = "http://inference-service:8000"
	}

	maxUploadMB := 10
	if v := os.Getenv("MAX_UPLOAD_MB"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			maxUploadMB = n
		}
	}

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	timeoutSec := 30
	if v := os.Getenv("INFERENCE_TIMEOUT"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			timeoutSec = n
		}
	}

	return config{
		InferenceURL:     inferenceURL,
		MaxUploadBytes:   maxUploadMB * 1024 * 1024,
		MaxUploadMB:      maxUploadMB,
		Port:             port,
		InferenceTimeout: time.Duration(timeoutSec) * time.Second,
	}
}

// jsonError returns a structured JSON error response.
func jsonError(c *fiber.Ctx, status int, errMsg, detail string) error {
	return c.Status(status).JSON(fiber.Map{
		"error":  errMsg,
		"detail": detail,
	})
}

// healthHandler responds to gateway health checks.
func healthHandler(c *fiber.Ctx) error {
	return c.JSON(fiber.Map{"status": "ok"})
}

// classifyHandler proxies image classification requests to the inference service.
func classifyHandler(cfg config) fiber.Handler {
	client := &http.Client{Timeout: cfg.InferenceTimeout}

	return func(c *fiber.Ctx) error {
		// Parse the uploaded file.
		fileHeader, err := c.FormFile("file")
		if err != nil {
			log.Printf("[WARN] No file in request: %v", err)
			return jsonError(c, fiber.StatusBadRequest, "missing file", "multipart field 'file' is required")
		}

		// Check file size.
		if fileHeader.Size > int64(cfg.MaxUploadBytes) {
			return jsonError(c, fiber.StatusRequestEntityTooLarge, "file too large",
				fmt.Sprintf("max upload size is %d MB", cfg.MaxUploadMB))
		}

		// Open the uploaded file.
		src, err := fileHeader.Open()
		if err != nil {
			log.Printf("[ERROR] Failed to open uploaded file: %v", err)
			return jsonError(c, fiber.StatusBadRequest, "bad request", "failed to read uploaded file")
		}
		defer src.Close()

		// Read the entire file into memory for MIME sniffing and forwarding.
		fileBytes, err := io.ReadAll(src)
		if err != nil {
			log.Printf("[ERROR] Failed to read uploaded file: %v", err)
			return jsonError(c, fiber.StatusBadRequest, "bad request", "failed to read uploaded file")
		}

		// MIME-sniff the first 512 bytes.
		sniffBytes := fileBytes
		if len(sniffBytes) > 512 {
			sniffBytes = sniffBytes[:512]
		}
		mimeType := http.DetectContentType(sniffBytes)

		if !strings.HasPrefix(mimeType, "image/") {
			return jsonError(c, fiber.StatusUnsupportedMediaType, "unsupported media type",
				fmt.Sprintf("expected image/*, got %s", mimeType))
		}

		// Also enforce size after reading (covers edge cases).
		if len(fileBytes) > cfg.MaxUploadBytes {
			return jsonError(c, fiber.StatusRequestEntityTooLarge, "file too large",
				fmt.Sprintf("max upload size is %d MB", cfg.MaxUploadMB))
		}

		// Build a new multipart request to forward to the inference service.
		var body bytes.Buffer
		writer := multipart.NewWriter(&body)

		part, err := writer.CreateFormFile("file", fileHeader.Filename)
		if err != nil {
			log.Printf("[ERROR] Failed to create multipart field: %v", err)
			return jsonError(c, fiber.StatusInternalServerError, "internal error", "failed to build proxy request")
		}

		if _, err := io.Copy(part, bytes.NewReader(fileBytes)); err != nil {
			log.Printf("[ERROR] Failed to copy file to multipart: %v", err)
			return jsonError(c, fiber.StatusInternalServerError, "internal error", "failed to build proxy request")
		}

		if err := writer.Close(); err != nil {
			log.Printf("[ERROR] Failed to close multipart writer: %v", err)
			return jsonError(c, fiber.StatusInternalServerError, "internal error", "failed to build proxy request")
		}

		// Create the HTTP request to the inference service.
		reqURL := cfg.InferenceURL + "/internal/predict"
		req, err := http.NewRequest(http.MethodPost, reqURL, &body)
		if err != nil {
			log.Printf("[ERROR] Failed to create request to inference: %v", err)
			return jsonError(c, fiber.StatusInternalServerError, "internal error", "failed to build proxy request")
		}
		req.Header.Set("Content-Type", writer.FormDataContentType())

		// Execute the request.
		log.Printf("[INFO] Forwarding classify request to %s (%d bytes)", reqURL, len(fileBytes))
		resp, err := client.Do(req)
		if err != nil {
			log.Printf("[ERROR] Inference service error: %v", err)
			return jsonError(c, fiber.StatusBadGateway, "inference service unavailable", err.Error())
		}
		defer resp.Body.Close()

		// Check inference service status.
		if resp.StatusCode != http.StatusOK {
			log.Printf("[WARN] Inference returned status %d", resp.StatusCode)
			return jsonError(c, fiber.StatusBadGateway, "inference error",
				fmt.Sprintf("inference returned status %d", resp.StatusCode))
		}

		// Stream the response body back.
		respBody, err := io.ReadAll(resp.Body)
		if err != nil {
			log.Printf("[ERROR] Failed to read inference response: %v", err)
			return jsonError(c, fiber.StatusBadGateway, "inference error", "failed to read inference response")
		}

		c.Set("Content-Type", "application/json")
		return c.Status(fiber.StatusOK).Send(respBody)
	}
}

func main() {
	cfg := loadConfig()

	log.Printf("[INFO] ML Gateway starting")
	log.Printf("[INFO] Inference URL: %s", cfg.InferenceURL)
	log.Printf("[INFO] Max upload: %d MB", cfg.MaxUploadMB)
	log.Printf("[INFO] Inference timeout: %s", cfg.InferenceTimeout)

	app := fiber.New(fiber.Config{
		BodyLimit:    cfg.MaxUploadBytes,
		ErrorHandler: customErrorHandler,
	})

	app.Get("/healthz", healthHandler)
	app.Post("/api/v1/classify", classifyHandler(cfg))

	log.Fatal(app.Listen(":" + cfg.Port))
}

// customErrorHandler ensures all Fiber errors are returned as JSON.
func customErrorHandler(c *fiber.Ctx, err error) error {
	code := fiber.StatusInternalServerError
	if e, ok := err.(*fiber.Error); ok {
		code = e.Code
	}
	return c.Status(code).JSON(fiber.Map{
		"error":  http.StatusText(code),
		"detail": err.Error(),
	})
}
