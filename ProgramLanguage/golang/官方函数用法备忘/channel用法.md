### 语法

| 操作 |	nil channel |	closed channel	| not-closed | non-nil channel
|--|--|--|--|--|
| close |	panic |	panic |	成功 |  close |
| 写 | ch <-	| 一直阻塞	| panic	| 阻塞或成功写入数据|
| 读 |  <- ch	| 一直阻塞	| 读取对应类型零值	| 阻塞或成功读取数据|

注意事项：

- channel初始化
  - channel在使用前，需要初始化，否则永远阻塞
- 无缓存channel
  - 无缓存channel，消费方和发送方会同时工作或者同时阻塞
- 关闭channel，可能panic
  - 关闭未初始化的channle(nil)会panic
  - 重复关闭同一channel会panic
  - 向以关闭channel发送消息会panic
- 关闭的chanel
  - 从已关闭channel读取数据，不会panic，若存在数据，则可以读出未被读取的消息，若已被读出，则获取的数据为零值，可以通过ok-idiom的方式，判断channel是否关闭
  - channel的关闭操作，会产生广播消息，所有向channel读取消息的goroutine都会接受到消息

### channel在使用前需要初始化，否则会永远阻塞

```go
package main

import (
	"fmt"
	"time"
)

func main() {
	go func() {
		var ch chan int
		fmt.Println("send 1 to ch start")
		ch <- 1
		fmt.Println("send 1 to ch stop")
	}()

	for {
		time.Sleep(time.Second)
		fmt.Println(time.Now().String())
	}
}
```

运行结果

```bash
$ go run main.go
send 1 to ch start
2022-03-06 13:45:48.930344 +0800 CST m=+1.001291631
2022-03-06 13:45:49.932467 +0800 CST m=+2.003408936
# ...
```

### 无缓存channel，消费方和发送方会同时工作或者同时阻塞

- 向无缓存的channel中发送消息也会堵塞，直到有goroutine从channel中读取消息。
- 从无缓存的channel中读取消息会堵塞，直到有goroutine往channel中发送消息

代码

```go
package main

import (
	"fmt"
	"time"
)

func main() {
	ch := make(chan int)

	go func() {
		fmt.Println(time.Now(), "send 1 to ch start")
		ch <- 1
		fmt.Println(time.Now(), "send 1 to ch stop")
	}()

	go func() {
		time.Sleep(time.Second)
		fmt.Println(time.Now(), "receive 1 to from ch start")
		<-ch
		fmt.Println(time.Now(), "receive 1 to from ch stop")
	}()

	time.Sleep(time.Minute) //wait
}
```

运行

```bash
$ go run main.go
2022-03-06 13:50:10.435786 +0800 CST m=+0.000105273 send 1 to ch start
# 等了1秒后，消费方开始消费，发送发才发送成功
2022-03-06 13:50:11.436957 +0800 CST m=+1.001274007 receive 1 to from ch start
2022-03-06 13:50:11.437105 +0800 CST m=+1.001421638 send 1 to ch stop
2022-03-06 13:50:11.437202 +0800 CST m=+1.001519499 receive 1 to from ch stop
```

## 参考

- [golang channel 使用总结](http://litang.me/post/golang-channel/)
- [GOLANG漫谈之CHANNEL妙法](https://ustack.io/2019-10-04-Golang%E6%BC%AB%E8%B0%88%E4%B9%8Bchannel%E5%A6%99%E6%B3%95.html)
