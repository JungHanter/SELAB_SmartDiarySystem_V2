import requests
import timeit


def push_text(text):
    time_sec = timeit.default_timer()
    data = {'user_id': 'lhs', 'title': "It's SPRING (Almost)!!!!", 'location': 'korea', 'timestamp': time_sec, 'text': text}
    r = requests.post(url='http://203.253.23.7:8000/api/' + 'diary', json=data)
    print(r.json())

if __name__ == '__main__':
    sample = """
        Today started out AMAZING!!! I woke up to a thick blanket of snow outside my window. The perfectly white, fresh snow that's just right for a Snow-Brandon (that's like a snowman, only it's got hair that swoops into its dreamy eyes).

I leapt out of bed and ran down the stairs to confirm with my parents that there was no school. One look at my mom's face and I knew:

SNOW DAY!!!!!!! WOOHOO !!!!

I was so excited! A day at home was EXACTLY what I needed. True, I wouldn't see Brandon all day. But I'm an independent woman! I planned to kick back, binge-watch something awesome, text with Chloe and Zoey…

But then, my leisurely snow day was totally ruined when Mom broke the news that I had to babysit Brianna while she and Dad braved the snow to run some errands!

WHY would my mother think it's safe for HER to go out in these extreme conditions??!! She is old and frail!! She might have a senior moment and somebody will find her days from now, a Mom-sicle who wandered too far from our house and couldn't get back, like in the movie "Frozen" when Anna almost froze to death!!

(Okay, since it's just me and my diary, I'll admit I wasn't really worried about my mother. I wish she'd stay home so I didn't have to deal with Brianna.) 

But NOPE. Mom and Dad both bundled up, hopped into the car, and disappeared into the snowstorm, possibly never to be heard from again. I was pretty sure they were going to crash the van in a snowdrift and have to decide who was going to eat the other one to survive.

Back at our house, Brianna was galloping around me screaming, "Snowmen! Snowball fights! Snow angels! Snow fort!"

I just shook my head and went to chill out on the couch. I barely got one text written, though, before Brianna was in my face, waving her mittens around, trying to figure out how to get them on right.

She is HOPELESS! If I hadn't been there to get her sorted out, I SWEAR she would have ended up with mittens on her ears!!!!

Once she was all bundled up, she just stood there, staring at me. I ignored her and sent my text. But it's not very easy to ignore an over-excited six-year-old. It's kind of like sitting on a beach and ignoring a tsunami.

"FINE!!" I finally said. "We can build a snowman and then we are coming inside for the rest of the day!!"

(In my mind, it was totally going to be a Snow-Brandon, but Brianna didn't have to know that.)

I should have gotten that agreement in writing, but Brianna can barely spell her name. We built a snowman AND made snow angels and THEN she somehow convinced me to build an igloo.

AN IGLOO!!!

So, I built the igloo for Brianna.  And, before I could even finish, she was already begging me to let her go inside it.

When I finally got done with the igloo, a super-excited Brianna got down on her hands and knees and crawled inside. She smiled and waved at me.  "Look at me!  I have a beautiful snow house just like Elsa!" she chirped.

That's when I decided to run inside the house and grab my camera so I could take a photo of Brianna.

But then, Brianna made a BIG MISTAKE and tried to stand up inside of the igloo.  Suddenly, her head popped through the top of it like a Jack-in-the-Box, and the whole thing collapsed all around her.  OMG!!  She looked like somebody had buried her from the neck down in the sand on a beach!

Brianna giggled nervously and said, "Oops!  My bad!"

There's more to this story, but my hands are still half-frozen from digging Brianna out of the igloo and I have to take a break from writing.  But, I'll finish telling you about my snow day with Brianna really soon!!!
    """  # from http://dorkdiaries.com/category/dorkworld/mydiary/page/3/
    push_text(sample)