word原本的定义代表CPU数据寄存器（data register）的大小。

对于8086而言，word的长度是2bytes，因为AX是16bit的。

而且word这个概念是在8086诞生的时候发明的。

![](/static/images/2412/p008.png)

但对于80386而言，尽管data register（eg. EAX）已经是32bit的，但word的长度就不在跟着修改了，仍然保留在了2bytes上。

所以当前讨论word就需要淡化它原本的含义，以及word的这个术语也就不再重要了。