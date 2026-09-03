代码库：https://github.com/zhyoulun/pcasm-code/blob/main/linux-ex/first.asm

linux必须在i386上编译，可以使用debian 12 i386版本

配合使用VS Code SFTP插件，可以在mac上远程编辑i386架构下的文件

编译

```bash
# 生成asm_io.o文件
nasm -f elf -d ELF_TYPE asm_io.asm

# 生成first.o文件
nasm -f elf first.asm

# 生成first文件
gcc -o first first.o driver.c asm_io.o
```

运行first可执行文件

```
zyl@debian:~/codes/zhyoulun/pcasm-code/linux-ex$ ./first
Enter a number: 1
Enter another number: 2
Register Dump # 1
EAX = 00000003 EBX = 00000003 ECX = 1782F538 EDX = BFA6FD20
ESI = 00412EDC EDI = B7FFEB80 EBP = BFA6FCD8 ESP = BFA6FCB8
EIP = 004101F7 FLAGS = 0206                PF
Memory Dump # 2 Address = 00413044
00413040 72 3A 20 00 59 6F 75 20 65 6E 74 65 72 65 64 20 "r: ?You entered "
00413050 00 20 61 6E 64 20 00 2C 20 74 68 65 20 73 75 6D "? and ?, the sum"
You entered 1 and 2, the sum of these is 3
```

其中first.asm的代码和相关注释，其中asm_io.inc是一个外部工具依赖

```
%include "asm_io.inc"


;
; initialized data is put in the .data segment
;
segment .data
;
; These labels refer to strings used for output
;
prompt1 db    "Enter a number: ", 0       ; don't forget nul terminator
prompt2 db    "Enter another number: ", 0
outmsg1 db    "You entered ", 0
outmsg2 db    " and ", 0
outmsg3 db    ", the sum of these is ", 0



;
; uninitialized data is put in the .bss segment
;
segment .bss
;
; These labels refer to double words used to store the inputs
;
input1  resd 1
input2  resd 1

 

;
; code is put in the .text segment
;
segment .text
        global  asm_main
asm_main:
        enter   0,0               ; setup routine
        pusha
;;;;;Enter a number: 1
        mov     eax, prompt1      ; print out prompt
        call    print_string

        call    read_int          ; read integer
        mov     [input1], eax     ; store into input1

;;;;;Enter another number: 2
        mov     eax, prompt2      ; print out prompt
        call    print_string

        call    read_int          ; read integer
        mov     [input2], eax     ; store into input2

        mov     eax, [input1]     ; eax = dword at input1
        add     eax, [input2]     ; eax += dword at input2
        mov     ebx, eax          ; ebx = eax
;;;;;Register Dump...
        dump_regs 1               ; dump out register values
;;;;;Memory Dump....
        dump_mem 2, outmsg1, 1    ; dump out memory
;
; next print out result message as series of steps
;
;;;;;You entered 1 and 2, the sum of these is 3
        mov     eax, outmsg1
        call    print_string      ; print out first message
        mov     eax, [input1]     
        call    print_int         ; print out input1
        mov     eax, outmsg2
        call    print_string      ; print out second message
        mov     eax, [input2]
        call    print_int         ; print out input2
        mov     eax, outmsg3
        call    print_string      ; print out third message
        mov     eax, ebx
        call    print_int         ; print out sum (ebx)
        call    print_nl          ; print new-line

        popa
        mov     eax, 0            ; return back to C
        leave                     
        ret
```

可以使用如下命令生成first.list文件，文件会列出hex文件和代码的对应关系

```bash
nasm -f elf -l first.list first.asm
```