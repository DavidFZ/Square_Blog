---
title: hash
date: 2022-09-08 14:01:49
tags:
---

# HASH
``` markdown
hashCode() to index conversion. To use hashCode() results as an index, we must convert the hashCode() to a valid index. Modulus does not work since hashCode may be negative. Taking the absolute value then the modulus also doesn’t work since Math.abs(Integer.MIN_VALUE) is negative. Typical approach: use hashCode & 0x7FFFFFFF instead before taking the modulus.

hashCode() 转换成有效的下标值的过程中，取模操作可能无法起作用。如果使用Math.abs方法后取得的值依旧是负值，取绝对值操作可能依旧无法起作用。标准的操作可以是使用 hasCode()& 0x7FFFFFFF 操作后再取模

There are no negative buckets so to avoid this you can remove the sign bit (the highest bit) and one way of doing this is to use a mask e.g. x & 0x7FFFFFFF which keeps all the bits except the top one. Another way to do this is to shift the output x >>> 1 however this is slower.

hasCode()& 0x7FFFFFFF 是为了取正
x >>> 1 可以做到相同的功能，但是会更慢
```