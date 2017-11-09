# Random Study Notes
This repository has a bunch of problems which I've been (weirdly) obsessed about. Upon digging into and solving (at least partially) them, I've also learned things (useful or not). Here it is, a collection of my study notes!
### Contents
[关于破解《伊岚翠》的字体乱码加密](关于破解《伊岚翠》的字体乱码加密.md): Solve the Mojibake problem in a chm file.
  * Unpack encrypted font file from chm;
  * Perl decompress eot file to ttf font;
  * Checkout cmap in ttf file (open source format);
  * 1 encrypted code -> 1 character in word with encrypted font (Alt+X) -> 1 character in pdf -> 1 OCR Unicode character -> 1 Unicode
  * VBA decrypt in MS Word

[浅谈pdf乱码](浅谈pdf乱码.md): Solve the Mojibake problem in a pdf file.
  * pdf digital structure
  * pdf logical structure (with pdftk)
  * pdf decode process, from CID to GID
  * ToUnicode map

[swf-pdf-doc88](swf-pdf-doc88.ipynb)
  * Handle binary file in python
  * zlib decompression in python
  * swf header structure

[Windows-10-wifi-auto-logon](Windows-10-wifi-auto-logon.md)
  * NCSI and its log in Windows Vista and above
  * Windows event filter in XML format
  * Task scheduler in Windows 10
  * NetworkProfile log
  * Connect wifi in Windows cmd

[Human-Resource-Machine](Human-Resource-Machine.md)
  * Implementing basic algorithms with assembly language
  * Optimization of assembly language

Fix for using juicer HiCCUPS tool with glibc 2.12
  * Come to this problem when trying to use juicer HiCCUPS tool on school's cluster, which has GNU libc at version 2.12. The following error

        java.lang.UnsatisfiedLinkError: /tmp/libJCudaRuntime-0.8.0-linux-x86_64.so: /lib64/libc.so.6: version `GLIBC_2.14' not found (required by /tmp/libJCudaRuntime-0.8.0-linux-x86_64.so)
    pops up, similar to the [this closed issue](https://github.com/jcuda/jcuda-main/issues/10). The problem is solved with the following steps:
    1. Extract jcuda native libraries build with glib 2.12 from [this zip file](files/HiCCUPS_glib2.12_patch.zip) and put it under desired path (for example, /path/to/jcuda)
    2. Use the HiCCUPS with jcuda native libraries above specified by `-D` option. Pre-load module and library if needed. For example,
       ```bash
       ml cuda/8.0
       export LD_PRELOAD=/usr/lib64/nvidia/libcuda.so.1
       java -Djava.library.path=/path/to/jcuda/lib64/ -jar /path/to/juicer_tools.1.7.6_jcuda.0.8.jar hiccups local/folder/HIC006.hic local/folder/hiccups_results
       ```
