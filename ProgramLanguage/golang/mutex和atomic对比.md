### Mutex vs Atomic

解决 race 的问题时，无非就是上锁。可能很多人都听说过一个高逼格的词叫「无锁队列」。 都一听到加锁就觉得很 low，那无锁又是怎么一回事？其实就是利用 atomic 特性，那 atomic 会比 mutex 有什么好处呢？go race detector 的作者总结了这两者的一个区别：

```
Mutexes do no scale. Atomic loads do.
```

mutex 由操作系统实现，而 atomic 包中的原子操作则由底层硬件直接提供支持。在 CPU 实现的指令集里，有一些指令被封装进了 atomic 包，这些指令在执行的过程中是不允许中断（interrupt）的，因此原子操作可以在 lock-free 的情况下保证并发安全，并且它的性能也能做到随 CPU 个数的增多而线性扩展。

若实现相同的功能，后者通常会更有效率，并且更能利用计算机多核的优势。所以，以后当我们想并发安全的更新一些变量的时候，我们应该优先选择用 atomic 来实现。

## 参考

- [谈谈 Golang 中的 Data Race](https://ms2008.github.io/2019/05/12/golang-data-race/)
