"""This shows an example of how to use the Generator object to provide more control within loops.
This could be used for things such as progress bars or emergency stops."""

from mandelpy import Settings,  Generator

s = Settings()

with Generator(s) as g:
    blocks = g.blocks
    for i, block in enumerate(blocks):
        print(f"Creating block {i + 1} of {len(blocks)}:", block)
        g.run_block(block)

    img = g.finished_img()

