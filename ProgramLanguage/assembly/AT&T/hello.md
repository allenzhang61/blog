.globl means that the assembler shouldn’t discard this symbol after assembly

_start is a special symbol that always needs to be marked with .globl because it marks the location of the start of the program.

Without marking this location in this way, when the computer loads your program it won’t know where to begin running your program.


we don’t have a .globl declaration for data_items. This is because we only refer to these locations within the program. No other file or program needs to know where they are located. This is in contrast to the _start symbol, which Linux needs to know where it is so that it knows where to begin the program’s execution. It’s not an error to write .globl data_items, it’s just not necessary.

- je: Jump if the values were equal
- jg: Jump if the second value was greater than the first value12
- jge: Jump if the second value was greater than or equal to the first value
- jl: Jump if the second value was less than the first value
- jle: Jump if the second value was less than or equal to the first value
- jmp: Jump no matter what. This does not need to be preceeded by a comparison.

incl increments the value of %edi by one

The way that the variables are stored and the parameters and return values are transferred by the computer varies from language to language as well. This variance is known as a language’s calling convention, because it describes how functions expect to get and receive data when they are called.

When we put the -lc on the command to link the helloworld program, it told the linker to use the c library (libc.so) to look up any symbols that weren’t already defined in helloworld.o.

However, it doesn’t actually add any code to our program, it just notes in the program where to look. When the helloworld program begins, the file /lib/ld-linux.so.2 is loaded first. This is the dynamic linker. This looks at our helloworld program and sees that it needs the c library to run. So, it searches for a file called libc.so in the standard places (listed in /etc/ld.so.conf and in the contents of the LD_LIBRARY_PATH environment variable), then looks in it for all the needed symbols (printf and exit in this case), and then loads the library into the program’s virtual memory. Finally, it replaces all instances of printf in the program with the actual location of printf in the library.

