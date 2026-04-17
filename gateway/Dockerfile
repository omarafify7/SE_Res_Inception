# Build stage
FROM golang:1.22-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY *.go ./
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o gateway .

# Runtime stage
FROM alpine:3.20
RUN apk --no-cache add wget
WORKDIR /app
COPY --from=builder /app/gateway .
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -qO- http://localhost:8080/healthz || exit 1
ENTRYPOINT ["/app/gateway"]
