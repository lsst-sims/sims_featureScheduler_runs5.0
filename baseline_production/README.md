Refactoring baseline.py to be ready for production.

* Looks like there is a slight difference compared to baseline due to change in alt az shadow mask size. 

* Having surveys ignore ToO observations so rolling cadence doesn't get as disturbed

* Includes mask of north in the morning for 3 hours. 

baseline 5.0.0 had 32k ToO observations. Now up to 88k. (or 1.6% to 4.2% of visits). Looks like both had a total of 1327 ToO events.


