### recover 只有在 defer 中调用才会生效

无效的例子

```go
package main

import "fmt"

func myRecover() {
	if r := recover(); r != nil {
		fmt.Println("recovered:", r)
	}

}

func main() {
	defer func() {
		myRecover()
	}()
	panic("this panic is not recovered")
}
```

运行结果

```bash
$ go run main.go
panic: this panic is not recovered

goroutine 1 [running]:
main.main()
	/tmp/sandbox2590866284/prog.go:18 +0x49

Program exited.
```

## 参考

- [5.4 panic 和 recover](https://draveness.me/golang/docs/part2-foundation/ch05-keyword/golang-panic-recover/)
