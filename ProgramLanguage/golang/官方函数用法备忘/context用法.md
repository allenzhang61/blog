## overview

Package context defines the Context type, which carries deadlines, cancellation signals, and other request-scoped values across API boundaries and between processes.

Incoming requests to a server should create a Context, and outgoing calls to servers should accept a Context. The chain of function calls between them must propagate the Context, optionally replacing it with a derived Context created using WithCancel, WithDeadline, WithTimeout, or WithValue. When a Context is canceled, all Contexts derived from it are also canceled.

The WithCancel, WithDeadline, and WithTimeout functions take a Context (the parent) and return a derived Context (the child) and a CancelFunc. Calling the CancelFunc cancels the child and its children, removes the parent's reference to the child, and stops any associated timers. Failing to call the CancelFunc leaks the child and its children until the parent is canceled or the timer fires. The go vet tool checks that CancelFuncs are used on all control-flow paths.

Programs that use Contexts should follow these rules to keep interfaces consistent across packages and enable static analysis tools to check context propagation:

Do not store Contexts inside a struct type; instead, pass a Context explicitly to each function that needs it. The Context should be the first parameter, typically named ctx:

```golang
func DoSomething(ctx context.Context, arg Arg) error {
	// ... use ctx ...
}
```

Do not pass a nil Context, even if a function permits it. Pass context.TODO if you are unsure about which Context to use.

Use context Values only for request-scoped data that transits processes and APIs, not for passing optional parameters to functions.

The same Context may be passed to functions running in different goroutines; Contexts are safe for simultaneous use by multiple goroutines.

## 几个context.WithXX用法

### 使用context.WithCanel()

子函数work()所在的协程先执行完

```go
package main

import (
	"context"
	"log"
	"time"
)

func work(ctx context.Context) {
	select {
	case <-ctx.Done():
		log.Printf("work func, ctx.Done, err: %+v\n", ctx.Err())
	case <-time.After(1 * time.Second):
		log.Printf("work success\n")
	}
}

func main() {
	ctx, cancel := context.WithCancel(context.Background())
	go work(ctx)

	time.Sleep(2 * time.Second)
	cancel()

	log.Printf("main func, ctx err: %+v\n", ctx.Err())
	time.Sleep(time.Second)
}
```

```
2021/04/04 20:55:23 work success
2021/04/04 20:55:24 main func, ctx err: context canceled
```

main()函数所在的协程先执行完

```go
package main

import (
	"context"
	"log"
	"time"
)

func work(ctx context.Context) {
	select {
	case <-ctx.Done():
		log.Printf("work func, ctx.Done, err: %+v\n", ctx.Err())
	case <-time.After(3 * time.Second):
		log.Printf("work success\n")
	}
}

func main() {
	ctx, cancel := context.WithCancel(context.Background())
	go work(ctx)

	time.Sleep(2 * time.Second)
	cancel()

	log.Printf("main func, ctx err: %+v\n", ctx.Err())
	time.Sleep(time.Second)
}
```

```
2021/04/04 20:56:34 main func, ctx err: context canceled
2021/04/04 20:56:34 work func, ctx.Done, err: context canceled
```

### context.WithTimeout && context.WithDeadline

两个函数的关系

```go
func WithTimeout(parent Context, timeout time.Duration) (Context, CancelFunc) {
	return WithDeadline(parent, time.Now().Add(timeout))
}
```

子函数work()所在的协程先执行完

```go
package main

import (
	"context"
	"log"
	"time"
)

func work(ctx context.Context) {
	select {
	case <-ctx.Done():
		log.Printf("work func, ctx.Done, err: %+v\n", ctx.Err())
	case <-time.After(1 * time.Second):
		log.Printf("work success\n")
	}
}

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	go work(ctx)

	select {
	case <-ctx.Done():
		log.Printf("main func, ctx Done, err: %+v\n", ctx.Err())
	}

	time.Sleep(time.Second)
}
```

```
2021/04/04 20:59:26 work success
2021/04/04 20:59:27 main func, ctx Done, err: context deadline exceeded
```

main()函数所在的协程先执行完

```go
package main

import (
	"context"
	"log"
	"time"
)

func work(ctx context.Context) {
	select {
	case <-ctx.Done():
		log.Printf("work func, ctx.Done, err: %+v\n", ctx.Err())
	case <-time.After(3 * time.Second):
		log.Printf("work success\n")
	}
}

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	go work(ctx)

	select {
	case <-ctx.Done():
		log.Printf("main func, ctx Done, err: %+v\n", ctx.Err())
	}

	time.Sleep(time.Second)
}
```

```
2021/04/04 21:00:18 main func, ctx Done, err: context deadline exceeded
2021/04/04 21:00:18 work func, ctx.Done, err: context deadline exceeded
```

### context.WithValue

```go
package main

import (
	"context"
	"fmt"
)

func main() {
	ctx := context.Background()
	ctx = context.WithValue(ctx, "a", 1)
	fmt.Println(ctx.Value("a"))
}
```

运行内容为1

## 参考

- [6.1 上下文 Context](https://draveness.me/golang/docs/part3-runtime/ch06-concurrency/golang-context/)
- [https://golang.org/pkg/context/](https://golang.org/pkg/context/)
