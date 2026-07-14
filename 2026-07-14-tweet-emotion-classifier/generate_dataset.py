"""
generate_dataset.py
--------------------
Builds a synthetic but realistic dataset of short, tweet-style sentences
labeled with one of six core emotions: joy, sadness, anger, fear, surprise, love.

Why synthetic data? This project is meant to run fully offline with no API
keys or downloads. Instead of scraping real tweets (which requires API access
and raises licensing/privacy concerns), we generate varied, natural-sounding
tweet-like text by combining hand-written phrase templates with word banks
for each emotion. Randomized combination + a fixed seed keeps the dataset
diverse but fully reproducible.

Run: python generate_dataset.py
Output: data/tweets_emotions.csv
"""

import csv
import random

random.seed(42)

# ---------------------------------------------------------------------------
# Word banks: each emotion gets subjects/triggers, feeling words, and emojis
# / hashtags that people commonly use on social media for that emotion.
# ---------------------------------------------------------------------------

TRIGGERS = {
    "joy": [
        "got the job offer", "finally finished my degree", "my team won the championship",
        "just adopted a puppy", "booked my dream vacation", "got promoted at work",
        "my favorite band announced a tour", "found $20 in my old jacket",
        "aced my final exam", "my best friend is visiting this weekend",
        "the sun is out after a week of rain", "just tried the best pizza ever",
        "my plant finally bloomed", "got tickets to the concert",
    ],
    "sadness": [
        "my flight got cancelled again", "lost my favorite necklace",
        "my childhood pet passed away", "didn't get the internship",
        "my best friend is moving away", "spilled coffee all over my laptop",
        "failed my driving test", "the show got cancelled after one season",
        "missed my grandma's birthday call", "my phone screen shattered",
        "it's been raining for a week straight", "forgot an important anniversary",
    ],
    "anger": [
        "my flight got delayed for the third time", "someone parked in my spot again",
        "the internet has been down all day", "my order arrived completely broken",
        "customer service hung up on me twice", "my neighbor's music at 3am",
        "the referee made an awful call", "prices went up again for no reason",
        "my package was marked delivered but never came", "traffic made me an hour late",
        "the meeting could have been an email", "my landlord ignored my messages for weeks",
    ],
    "fear": [
        "there's a huge spider in my room", "I have a big presentation tomorrow",
        "the plane hit turbulence out of nowhere", "I heard a strange noise downstairs",
        "my exam results come out tomorrow", "walking home alone in the dark",
        "the doctor wants to run more tests", "the storm warning just got upgraded",
        "I might lose my job in the next round of layoffs", "the power went out during the thunderstorm",
        "I have to speak in front of the whole company", "my car started making a weird sound on the highway",
    ],
    "surprise": [
        "my friends threw me a surprise party", "I bumped into my old teacher at the airport",
        "the plot twist in that finale", "I won a raffle I forgot I entered",
        "my package arrived two weeks early", "I found out my coworker used to be a pro athlete",
        "the sequel is actually better than the original", "my quiet neighbor turned out to be a famous author",
        "I got an email saying my rent is dropping", "the store had my exact size in stock",
        "my phone battery lasted two full days", "the weather forecast was actually right for once",
    ],
    "love": [
        "spent the whole day with my partner", "my mom sent me a care package",
        "watching my dog get excited when I come home", "my grandparents' 50th anniversary",
        "a random stranger paid for my coffee", "my friends drove three hours just to see me",
        "rewatching our wedding video", "my sister remembered my favorite snack",
        "holding hands during the sunset walk", "my partner cooked my favorite meal",
        "getting a handwritten letter from an old friend", "cuddling with my cat on a rainy day",
    ],
}

FEELING_PHRASES = {
    "joy": [
        "I am so happy right now", "this made my whole week", "I can't stop smiling",
        "feeling absolutely thrilled", "best day ever honestly", "I'm overjoyed",
        "pure happiness right here", "living my best life today",
    ],
    "sadness": [
        "I feel so down about it", "this really broke my heart", "I can't stop feeling blue",
        "it's making me tear up", "feeling pretty low today", "I'm heartbroken honestly",
        "just feels like a gloomy day", "I could really use a hug right now",
    ],
    "anger": [
        "I am absolutely furious", "this is so frustrating", "I'm fuming right now",
        "makes my blood boil", "I've had it with this", "so annoyed I could scream",
        "this is completely unacceptable", "I'm seething about it",
    ],
    "fear": [
        "I'm terrified honestly", "my heart is racing", "I'm so nervous about it",
        "feeling really anxious right now", "this is freaking me out", "I can't shake this dread",
        "I'm scared out of my mind", "my hands won't stop shaking",
    ],
    "surprise": [
        "I did not see that coming", "I'm completely shocked", "wow I was not expecting this",
        "this caught me totally off guard", "I'm speechless right now", "no way this actually happened",
        "I still can't believe it", "what a plot twist honestly",
    ],
    "love": [
        "my heart is so full", "I feel so loved right now", "this warms my heart every time",
        "I adore this so much", "feeling so grateful and loved", "this is what love looks like",
        "my heart could burst", "I cherish moments like this",
    ],
}

TAGS = {
    "joy": ["#blessed", "#happy", "#goodvibes", "#grateful", ""],
    "sadness": ["#sad", "#heartbroken", "#notokay", ""],
    "anger": ["#furious", "#done", "#overit", ""],
    "fear": ["#anxious", "#scared", "#nervous", ""],
    "surprise": ["#shocked", "#nooway", "#wow", ""],
    "love": ["#love", "#grateful", "#blessed", ""],
}

EMOJIS = {
    "joy": ["😄", "🎉", "😁", "🙌"],
    "sadness": ["😢", "💔", "😞", "😔"],
    "anger": ["😡", "🤬", "😤", ""],
    "fear": ["😨", "😰", "😬", ""],
    "surprise": ["😲", "😳", "🤯", "😮"],
    "love": ["❤️", "🥰", "😍", "🤗"],
}

TEMPLATES = [
    "Just {trigger} and {feeling} {emoji} {tag}",
    "{feeling_cap} because {trigger}. {emoji} {tag}",
    "So {trigger}... {feeling} {emoji} {tag}",
    "Can we talk about how {trigger}? {feeling} {emoji}",
    "{feeling_cap}. {trigger_cap}. {tag}",
    "Not going to lie, {trigger} and {feeling} {emoji}",
    "{trigger_cap} today and honestly {feeling} {tag}",
]


def build_tweet(emotion: str) -> str:
    """Randomly assemble one tweet-like sentence for the given emotion."""
    trigger = random.choice(TRIGGERS[emotion])
    feeling = random.choice(FEELING_PHRASES[emotion])
    emoji = random.choice(EMOJIS[emotion])
    tag = random.choice(TAGS[emotion])
    template = random.choice(TEMPLATES)

    text = template.format(
        trigger=trigger,
        trigger_cap=trigger[0].upper() + trigger[1:],
        feeling=feeling,
        feeling_cap=feeling[0].upper() + feeling[1:],
        emoji=emoji,
        tag=tag,
    )
    # Clean up double spaces left behind by empty emoji/tag slots
    text = " ".join(text.split())
    return text


def main(n_per_class: int = 60, out_path: str = "data/tweets_emotions.csv"):
    rows = []
    seen = set()
    for emotion in TRIGGERS:
        count = 0
        attempts = 0
        while count < n_per_class and attempts < n_per_class * 20:
            attempts += 1
            tweet = build_tweet(emotion)
            if tweet in seen:
                continue
            seen.add(tweet)
            rows.append({"text": tweet, "emotion": emotion})
            count += 1

    random.shuffle(rows)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "emotion"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} labeled tweets to {out_path}")
    counts = {}
    for r in rows:
        counts[r["emotion"]] = counts.get(r["emotion"], 0) + 1
    print("Class distribution:", counts)


if __name__ == "__main__":
    main()
