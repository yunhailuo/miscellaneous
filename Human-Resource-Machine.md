# Solutions for Human Resource Machine
[**human-resource-machine-solutions**](human-resource-machine-solutions) has all my solutions to the puzzle game "[Human Resource Machine](http://tomorrowcorporation.com/humanresourcemachine)". To my understanding, "Human Resource Machine" is a fun game simulating assembly programming. I really enjoy seeing how it works, knowing the restricted set of command and optimizing solutions for speed and/or size. Here, I record my exciting journey.

This collection is just my own solutions, not necessarily the best. For the best and the most comprehensive collection of solutions, I would go to [this repo](https://github.com/atesgoral/hrm-solutions) and [its website](http://atesgoral.github.io/hrm-solutions/).
### Notes:
1. Two interesting ideas/concepts I've learned from others:
   * "[Loop unrolling](https://en.wikipedia.org/wiki/Loop_unrolling)": a space-time tradeoff which optimize speed at the expense of size. Learned from the [Steam forum](https://steamcommunity.com/app/375820/discussions/0/483367798502748659/) and used in the [solution for year 2](human-resource-machine-solutions/02-Busy-Mail-Room-5.25.txt)
   * Optimize speed by jumping back (save one jump step) and skipping the first output: learned from [another solution](https://github.com/atesgoral/hrm-solutions/blob/master/solutions/09-Zero-Preservation-Initiative-5.25/5.25-nanashi-juanto.asm) and used in solutions for [year 9](human-resource-machine-solutions/09-Zero-Preservation-Initiative-5.25.txt), [year 13](human-resource-machine-solutions/13-Equalization-Room-9.27.txt) and [year 16]](human-resource-machine-solutions/16-Absolute-Positivity-8.34.txt). To understand the difference, compare the following two:
      * Straightforward code

            a:
            b:
                INBOX   
                JUMPZ    c
                JUMP     b
            c:
                OUTBOX  
                JUMP     a
      * Optimized code

               JUMP     b
            a:
                OUTBOX  
            b:
            c:
                INBOX   
                JUMPZ    a
                JUMP     c
        For each run, optimized code has one more `JUMP` than straightforward code. However, for each zero output, optimized code saves one `JUMP` by jumping up back directly. Therefore, as long as there is one zero output, optimized code won't be worse than straightforward code.
1. Selected interesting challenges

|Challenges|Notes|
|---|---|
|[Year 28 - size](human-resource-machine-solutions/28-Three-Sort-34.98.txt)|Pair-wise comparison for every new number; use the storage without swapping numbers.|
|[Year 40](human-resource-machine-solutions/40-Prime-Factory-27.355.txt)|Initialize a slot with "1" for odd number sequence after processing "2".|
|[Year 41](human-resource-machine-solutions/41-Sorting-Floor-27.714.txt)|Pair-wise comparison for every new number|
