```go
package main

import (
	"encoding/json"
	"log"
	"net"
	"net/http"
	"net/url"
)

func debugResponse(r *http.Request) []byte {
	s, _ := json.MarshalIndent(struct {
		Domain           string
		Host             string
		Method           string
		Proto            string
		ContentLength    int64
		TransferEncoding []string
		Close            bool
		RemoteAddr       string
		RequestURI       string
		URL              *url.URL
		Header           http.Header
	}{
		Host:             r.Host,
		Method:           r.Method,
		Proto:            r.Proto,
		URL:              r.URL,
		Header:           r.Header,
		ContentLength:    r.ContentLength,
		TransferEncoding: r.TransferEncoding,
		Close:            r.Close,
		RemoteAddr:       r.RemoteAddr,
		RequestURI:       r.RequestURI,
	}, "", "  ")
	return s
}

func customHandler(w http.ResponseWriter, r *http.Request) {
	log.Println(w.Write(debugResponse(r)))
}

func main() {
	ln, err := net.Listen("tcp", "127.0.0.1:8080")
	if err != nil {
		log.Fatalf("net Listen fail, err: %+v", err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/", customHandler)

	server := &http.Server{
		Addr:    "", //useless here
		Handler: mux,
	}
	log.Printf("server Serve return, err: %+v", server.Serve(ln))
}
```