

```go
package sync002

import (
	"sync"
	"sync/atomic"
	"testing"
)

func BenchmarkAtomic(b *testing.B) {
	var count int64 = 0

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			//基于atomic保证原子性
			atomic.AddInt64(&count, 1)
		}
	})

	if count != int64(b.N) {
		b.Errorf("bad result")
	}
}

func BenchmarkMutex(b *testing.B) {
	var count int64 = 0
	var countMutex = &sync.Mutex{}

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			//基于mutex保证原子性
			countMutex.Lock()
			count++
			countMutex.Unlock()
		}
	})

	if count != int64(b.N) {
		b.Errorf("bad result")
	}
}
```

运行

```
goos: darwin
goarch: amd64
cpu: Intel(R) Core(TM) i7-1068NG7 CPU @ 2.30GHz
BenchmarkAtomic
BenchmarkAtomic-8   	75652459	        17.32 ns/op
BenchmarkMutex
BenchmarkMutex-8    	20609574	        57.22 ns/op
PASS
```

## 参考

- [Benchmark: sync.RWMutex vs atomic.Value](https://gist.github.com/dim/152e6bf80e1384ea72e17ac717a5000a)
- [Go如何减少atomic包的锁冲突](https://double12gzh.github.io/2020/09/10/%E5%A6%82%E4%BD%95%E9%81%BF%E5%85%8DAtomic%E5%8C%85%E4%B8%AD%E7%9A%84%E9%94%81/)
