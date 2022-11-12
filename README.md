# best-wordle-double-opener
Figuring out what's the best two-word opener on the game wordle.  
Inspired by [Grant Anderson's video](https://youtu.be/v68zYyaEmEA) that tried to find the best opening word on wordle using information theory. I tried to use the method in the video to find the best two-word opener (the two words that when used at the beginning of the game will give the most information on average).    
This is much harder than finding the best single opener since the search space is now squared (number of allowed word is $12953$, meaning there are $12953 * 12952\approx 10^8$ possible word pairs). And for every word pair we have to check each outcome ( ${3^5}^2\approx 59049$ possible outcomes). 
I checked the most promising word pairs randomly and only managed to go through $\sim 0.3\\%$ of the search space. More progress can be made with more computing power or implementing multiprocessing.  
The current best two-word opener is `nicer` and `loast`, giving $\sim 9.47$ bit of information on average.
