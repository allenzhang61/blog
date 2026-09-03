## Wait函数备忘

```
func (c *Cond) Wait()
```

Wait()会自动释放c.L，并挂起调用者的goroutine。之后恢复执行，Wait()会在返回时对c.L加锁。

除非被Signal或者Broadcast唤醒，否则Wait()不会返回。

由于Wait()第一次恢复时，C.L并没有加锁，所以当Wait返回时，调用者通常并不能假设条件为真。

取而代之的是, 调用者应该在循环中调用Wait。（简单来说，只要想使用condition，就必须加锁。）

```
c.L.Lock()
for !condition() {
    c.Wait()
}
... make use of condition ...
c.L.Unlock()
```

### 使用示例

```go
package main

import (
	"fmt"
	"sync"
	"time"
)

var sharedRsc = false

func main() {
	var wg sync.WaitGroup
	wg.Add(2)
	m := sync.Mutex{}
	c := sync.NewCond(&m)
	go func() {
		// this go routine wait for changes to the sharedRsc
		c.L.Lock()
		for sharedRsc == false {
			fmt.Println("goroutine1 wait")
			c.Wait()
		}
		fmt.Println("goroutine1", sharedRsc)
		c.L.Unlock()
		wg.Done()
	}()

	go func() {
		// this go routine wait for changes to the sharedRsc
		c.L.Lock()
		for sharedRsc == false {
			fmt.Println("goroutine2 wait")
			c.Wait()
		}
		fmt.Println("goroutine2", sharedRsc)
		c.L.Unlock()
		wg.Done()
	}()

	// this one writes changes to sharedRsc
	time.Sleep(2 * time.Second)
	c.L.Lock()
	fmt.Println("main goroutine ready")
	sharedRsc = true
	c.Broadcast()
	fmt.Println("main goroutine broadcast")
	c.L.Unlock()
	wg.Wait()
}
```

运行结果

```
goroutine1 wait
goroutine2 wait
main goroutine ready
main goroutine broadcast
goroutine2 true
goroutine1 true
```


## 参考

- [Golang中如何正确使用条件变量sync.Cond](https://ieevee.com/tech/2019/06/15/cond.html)
