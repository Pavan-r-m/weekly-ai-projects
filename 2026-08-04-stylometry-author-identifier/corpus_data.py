"""
corpus_data.py
--------------
Small bundled demo corpus for the Stylometric Author Identifier.

Each entry is a (author_name, raw_text) excerpt taken from the opening
chapters of well-known PUBLIC DOMAIN novels (all first published more than
95 years ago, so they are in the public domain worldwide). The excerpts are
intentionally short so the whole project runs instantly with no downloads
and no API keys.

NOTE: This is a *demo* corpus meant to show the pipeline end-to-end. The
`author_identifier.py` tool works on any folder of author-labelled .txt
files, so you can drop in much larger corpora (e.g. full novels from
Project Gutenberg) for a more robust real-world model -- see the README.
"""

CORPUS = {
    "Herman Melville": [
        """Call me Ishmael. Some years ago -- never mind how long precisely --
        having little or no money in my purse, and nothing particular to
        interest me on shore, I thought I would sail about a little and see
        the watery part of the world. It is a way I have of driving off the
        spleen and regulating the circulation. Whenever I find myself growing
        grim about the mouth; whenever it is a damp, drizzly November in my
        soul; whenever I find myself involuntarily pausing before coffin
        warehouses, and bringing up the rear of every funeral I meet; and
        especially whenever my hypos get such an upper hand of me, that it
        requires a strong moral principle to prevent me from deliberately
        stepping into the street, and methodically knocking people's hats
        off -- then, I account it high time to get to sea as soon as I can.
        This is my substitute for pistol and ball. With a philosophical
        flourish Cato throws himself upon his sword; I quietly take to the
        ship. There is nothing surprising in this. If they but knew it,
        almost all men in their degree, some time or other, cherish very
        nearly the same feelings towards the ocean with me.""",
    ],
    "Jane Austen": [
        """It is a truth universally acknowledged, that a single man in
        possession of a good fortune, must be in want of a wife. However
        little known the feelings or views of such a man may be on his first
        entering a neighbourhood, this truth is so well fixed in the minds
        of the surrounding families, that he is considered as the rightful
        property of some one or other of their daughters. 'My dear Mr.
        Bennet,' said his lady to him one day, 'have you heard that
        Netherfield Park is let at last?' Mr. Bennet replied that he had
        not. 'But it is,' returned she; 'for Mrs. Long has just been here,
        and she told me all about it.' Mr. Bennet made no answer. 'Do you
        not want to know who has taken it?' cried his wife impatiently.
        'You want to tell me, and I have no objection to hearing it.' This
        was invitation enough.""",
    ],
    "Charles Dickens": [
        """It was the best of times, it was the worst of times, it was the
        age of wisdom, it was the age of foolishness, it was the epoch of
        belief, it was the epoch of incredulity, it was the season of
        Light, it was the season of Darkness, it was the spring of hope,
        it was the winter of despair, we had everything before us, we had
        nothing before us, we were all going direct to Heaven, we were all
        going direct the other way -- in short, the period was so far like
        the present period, that some of its noisiest authorities insisted
        on its being received, for good or for evil, in the superlative
        degree of comparison only.""",
        """My father's family name being Pirrip, and my Christian name
        Philip, my infant tongue could make of both names nothing longer or
        more explicit than Pip. So, I called myself Pip, and came to be
        called Pip. I give Pirrip as my father's family name, on the
        authority of his tombstone and my sister -- Mrs. Joe Gargery, who
        married the blacksmith. As I never saw my father or my mother, and
        never saw any likeness of either of them, my first fancies
        regarding what they were like were unreasonably derived from their
        tombstones.""",
    ],
    "Lewis Carroll": [
        """Alice was beginning to get very tired of sitting by her sister
        on the bank, and of having nothing to do: once or twice she had
        peeped into the book her sister was reading, but it had no pictures
        or conversations in it, 'and what is the use of a book,' thought
        Alice, 'without pictures or conversations?' So she was considering
        in her own mind, as well as she could, for the hot day made her
        feel very sleepy and stupid, whether the pleasure of making a
        daisy-chain would be worth the trouble of getting up and picking
        the daisies, when suddenly a White Rabbit with pink eyes ran close
        by her. There was nothing so very remarkable in that; nor did
        Alice think it so very much out of the way to hear the Rabbit say
        to itself, 'Oh dear! Oh dear! I shall be late!'""",
    ],
}

# A short "mystery" excerpt used by the demo to show live prediction.
# (Opening lines of Moby-Dick's third paragraph -- same author as above,
# but NOT included in the training data, so it is a genuine held-out test.)
MYSTERY_TEXT = """There now is your insular city of the Manhattoes, belted
round by wharves as Indian isles by coral reefs -- commerce surrounds it
with her surf. Right and left, the streets take you waterward. Its
extreme downtown is the battery, where that noble mole is washed by waves,
and cooled by breezes, which a few hours previous were out of sight of
land. Look at the crowds of water-gazers there. Circumambulate the city of
a dreamy Sabbath afternoon. Go from Corlears Hook to Coenties Slip, and
from thence, by Whitehall, northward. What do you see? -- Posted like
silent sentinels all around the town, stand thousands upon thousands of
mortal men fixed in ocean reveries."""
