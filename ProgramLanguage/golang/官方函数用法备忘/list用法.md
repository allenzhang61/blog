### 基本用法

```go
package main

import (
	"container/list"
	"fmt"
)

func main() {
	l := list.New()
	showList(l)
	l.PushBack(1)
	showList(l)
}

func showList(l *list.List) {
	fmt.Print("list: ")
	for e := l.Front(); e != nil; e = e.Next() {
		fmt.Print(e.Value.(int), "->")
	}
	fmt.Println("nil")
}
```

```bash
go run main.go
list: nil
list: 1->nil
```
