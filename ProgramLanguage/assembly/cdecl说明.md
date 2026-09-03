driver.c

```c
#include "cdecl.h"

int PRE_CDECL asm_main( void ) POST_CDECL;

int main()
{
    int ret_status;
    ret_status = asm_main();
    return ret_status;
}
```

cdecl.h

```c
#ifndef CDECL_HEADER_FILE
#define CDECL_HEADER_FILE

/*
 * Define macros to specify the standard C calling convention
 * The macros are designed so that they will work with all
 * supported C/C++ compilers.
 *
 * To use define your function prototype like this:
 *
 * return_type PRE_CDECL func_name( args ) POST_CDECL;
 *
 * For example:
 *
 * int PRE_CDECL f( int x, int y) POST_CDECL;
 */


#if defined(__GNUC__)
#  define PRE_CDECL
#  define POST_CDECL __attribute__((cdecl))
#else
#  define PRE_CDECL __cdecl
#  define POST_CDECL
#endif


#endif
```

`__attribute__((cdecl))` 是 GCC（GNU 编译器）提供的一种属性，用于明确指定函数使用 cdecl（C declaration）调用约定。调用约定（calling convention）定义了函数如何接收参数、如何返回值以及如何管理堆栈等。cdecl 是最常见的 C 语言调用约定之一，特别是在 x86 架构上。

`__attribute__((cdecl))` 用于指定函数遵循 cdecl 调用约定。在 GCC 中，cdecl 通常是默认的调用约定，因此显式指定这个属性通常不是必需的，但它可以确保代码的可读性和明确性。

语法

```
return_type function_name(arguments) __attribute__((cdecl));
```

示例

```
int add(int a, int b) __attribute__((cdecl));
```

在 cdecl 调用约定下，函数的参数是 从右到左 压入栈的。也就是说，函数的最后一个参数会首先被压入栈中，依此类推。

例如`add(2, 3);`，参数传递顺序为：3（第一个压栈），然后是 2（第二个压栈）。

使用 cdecl 调用约定时，调用者 负责清理堆栈。这意味着，调用函数后，调用者需要移除参数所占的堆栈空间。

```
int add(int a, int b) {
    return a + b;
}

int main() {
    int result = add(5, 3);  // 调用 add 函数
    return 0;
}
```

在这个例子中，main 函数（调用者）负责清理堆栈。

在 cdecl 调用约定下，返回值通常通过 EAX 寄存器（对于 32 位系统）返回。

示例

```c
#include <stdio.h>

int add(int a, int b) __attribute__((cdecl));

int add(int a, int b) {
    return a + b;
}

int main() {
    int result = add(5, 3);  // 调用 add 函数
    printf("Result: %d\n", result);
    return 0;
}
```

假设你需要使用汇编语言与 C 函数交互，以下是如何调用使用 cdecl 调用约定的函数的示例。

汇编代码

```
; 汇编代码：调用 C 函数 add
section .text
extern add
global _start

_start:
    ; 压栈参数
    push 5         ; 参数 b
    push 10        ; 参数 a
    call add       ; 调用 add 函数
    ; 结果将存储在 EAX 寄存器中
    ; 做一些操作（例如退出程序）
    mov ebx, 0
    mov eax, 1
    int 0x80
```

在这个汇编示例中，add 函数的调用遵循 cdecl 调用约定，参数按右到左的顺序压栈，调用结束后调用者负责清理堆栈。


默认行为：在 GCC 中，cdecl 通常是默认的调用约定，因此大多数情况下不需要显式地使用 `__attribute__((cdecl))`。但如果你希望明确指定调用约定，或者与其他遵循 cdecl 的代码交互时，使用这个属性是有意义的。

跨平台兼容性：确保你的代码在支持该属性的编译器上编译（如 GCC 或 Clang）。在其他编译器（例如 MSVC）中，`__attribute__((cdecl))` 可能不适用，需要使用其他方式指定调用约定。

性能：虽然 cdecl 是一个常见的调用约定，但在某些情况下，如果你需要优化函数调用和堆栈管理，可能会选择其他调用约定，如 stdcall 或 fastcall。