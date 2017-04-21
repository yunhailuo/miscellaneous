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

