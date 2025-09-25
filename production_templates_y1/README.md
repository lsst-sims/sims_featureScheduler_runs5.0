Try having a template gathering tier that only runs in year 1

* adding explicit "good seeing" limit of 1.3". Some basis functions where using a default of 0.8 before.
* swapping DDF dithering to per exposure rather than per night



pty1_okrepeat.py : Trying out removing the repeat-in-night suppression, because it seems like 
it shouldn't be relevant often and has a lot of magic numbers with it. But it does turn out to be helping SNe light curves quite a bit, so it probably should stay in there.

pty1_gs.py: turning the "good seeing" for the basis function back to 0.8". Check if there is a difference for galaxy shapes. Update docstrings and comments that it's there for galaxy shapes rather than templates.

