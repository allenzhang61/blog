### 摘自《http包源码解析》

1、首先调用Http.HandleFunc，按如下顺序执行：

- 调用了DefaultServerMux的HandleFunc。
- 调用了DefaultServerMux的Handle。
- 往DefaultServerMux的map[string] muxEntry中增加对应的handler和路由规则。

2、调用http.ListenAndServe(":9090",nil)，按如下顺序执行：

- 实例化Server。
- 调用Server的ListenAndServe()。
- 调用net.Listen("tcp",addr)监听端口。
- 启动一个for循环，在循环体中Accept请求。
- 对每个请求实例化一个Conn，并且开启一个goroutine为这个请求进行服务go c.serve()。
- 读取每个请求的内容w,err:=c.readRequest()。
- 判断handler是否为空，如果没有设置handler，handler默认设置为DefaultServeMux。
- 调用handler的ServeHttp。
- 根据request选择handler，并且进入到这个handler的ServeHTTP, mux.handler(r).ServeHTTP(w,r)
- 选择handler
- 判断是否有路由能满足这个request（循环遍历ServeMux的muxEntry）。
- 如果有路由满足，调用这个路由handler的ServeHttp。
- 如果没有路由满足，调用NotFoundHandler的ServeHttp。

## 参考

- [http包源码解析](https://www.huweihuang.com/golang-notes/web/golang-http-execution-flow.html)
