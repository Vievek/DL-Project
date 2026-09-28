# TextCNN error analysis (validation split only)

_Content warning: examples contain offensive language (dataset content)._

Model: final TextCNN (C4, GloVe), val macro-F1 0.6486. Thresholds tuned per class on validation (all cutoffs).

## Per-class errors

| label         |   val_positives |   TP |   FP |   FN |   precision |   recall |   threshold |
|:--------------|----------------:|-----:|-----:|-----:|------------:|---------:|------------:|
| toxic         |            2294 | 1807 |  500 |  487 |       0.783 |    0.788 |      0.8335 |
| severe_toxic  |             240 |  158 |  287 |   82 |       0.355 |    0.658 |      0.9643 |
| obscene       |            1267 | 1055 |  228 |  212 |       0.822 |    0.833 |      0.9591 |
| threat        |              71 |   44 |   42 |   27 |       0.512 |    0.62  |      0.9973 |
| insult        |            1182 |  949 |  489 |  233 |       0.66  |    0.803 |      0.9156 |
| identity_hate |             211 |  131 |  150 |   80 |       0.466 |    0.621 |      0.9964 |


## 10 false positives (clean comments, most confident wrong flags)

- **val row 16436** | true: clean | pred: toxic, obscene, insult | probs: toxic 1.0000, obscene 1.0000, insult 1.0000 | tokens: 196 (TRUNCATED) | OOV: none
  > JzG Wrong is still wrong, the circumstances don't matter. You won't be punished though because you are part of the admin. of this site now. You need to grow up, you fucking loser. I just called it how I saw it. It is a suicide and she will burn in hell for it. Also, You will never get rid of me, you …

- **val row 6153** | true: clean | pred: toxic, obscene, insult | probs: toxic 1.0000, obscene 0.9998, insult 0.9985 | tokens: 553 (TRUNCATED) | OOV: dmx, messin
  > [Chorus: DMX - repeat 2X] Where the hood, where the hood, where the hood at? Have that nigga in the cut, where the wood at? Oh, them niggaz actin up?!? Where the wolves at? You better BUST THAT if you gon pull that [DMX] Man, cats don't know what it's gonna be messin with a dog like me, D-to-the-M-t …

- **val row 10079** | true: clean | pred: toxic, obscene, insult | probs: toxic 0.9999, obscene 0.9976, insult 0.9976 | tokens: 53 | OOV: none
  > ALSO, you have allowed an ERRONEOUS link to stand, even though yu have been told, repeatedly, that it is false! The listed performer, Helen Hughes is OBVIOUSLY an ACTOR, not a PROFESSOR. Why have you not had this corrected, you blithering TWERP?!

- **val row 1367** | true: clean | pred: toxic, obscene, insult | probs: toxic 0.9998, obscene 0.9994, insult 0.9946 | tokens: 689 (TRUNCATED) | OOV: caffeinated, grilled, pecs, punchies, rigby's, vchip
  > Some ratings from the VCHip 1. The Power: TV-PG-L (hell, shit, pissed) 2. Just Set Up the Chairs: TV-PG-V (the Destroyer of World murders) 3. Caffeinated Concert Tickets: TV-PG-D (lady pecs = boobs, real reason people go to concerts is too see hot girls) 4. Death Punchies: TV-PG-V: (death kwon do, R …

- **val row 8819** | true: clean | pred: toxic, obscene, insult, identity_hate | probs: toxic 0.9997, obscene 0.9708, insult 0.9743, identity_hate 0.9995 | tokens: 974 (TRUNCATED) | OOV: none
  > "If you like to ""squash IP addresses for fun"", then you'll never earn my respect and should not be at Wikipedia. Keep it in mind when you decide that calling me a vandal will hide your transgressions. My idols are among the Christian right. If you hate them and erase my Talk Page as you did my Use …

- **val row 4681** | true: clean | pred: toxic, obscene, insult | probs: toxic 0.9997, obscene 0.9992, insult 0.9943 | tokens: 168 (TRUNCATED) | OOV: l1
  > " I hate to break it to you but being a ""mother tongue"" speaker of South African English doesn't give you any more credibility than anyone else, friend. Just because you speak a language or dialect doesn't mean you know shit about it. You don't have to know anything about a language to be able to  …

- **val row 682** | true: clean | pred: toxic, obscene, insult | probs: toxic 0.9997, obscene 0.9927, insult 0.9656 | tokens: 99 | OOV: dragqueen, hahahahahahaaa
  > " Yes yes and yes! I don't know the songs directly by name, but if it were to play, I know I'd dance my ass off cause songs like those are classics!!! D I was at a dragqueen bar this weekend, and ""Countdown"" came on, and me and my friend got up and danced all over the place! Hahahahahahaaa!!! I mi …

- **val row 23035** | true: clean | pred: toxic, obscene, insult | probs: toxic 0.9996, obscene 0.9991, insult 0.9931 | tokens: 164 (TRUNCATED) | OOV: valuble
  > " Fine, Triple F...I was having a bad day, and as most already know, I'm easily angered, so that didn't help the fact. But, let me say something here: Let's think for a few minutes. If AMIB had done this, no one would've done anything. The guy gets away with murder. He's constantly cursing his ass o …

- **val row 19323** | true: clean | pred: toxic, obscene, insult | probs: toxic 0.9996, obscene 0.9856, insult 0.9675 | tokens: 167 (TRUNCATED) | OOV: none
  > " Phil, if you want to send the message that you support Eleland's comments, why not just give him a barnstar? I quote from you in the AN/I thread linked above ""if he posts a conciliatory unblock request that should be granted"". Where is that statement? In fact, he reiterates that he continues to  …

- **val row 11996** | true: clean | pred: toxic, obscene, insult | probs: toxic 0.9996, obscene 0.9984, insult 0.9925 | tokens: 157 (TRUNCATED) | OOV: bock, haber, hardass, localzuk
  > See? He understands, and doesn't want interference or any trouble. He even censored my comments to toe the Party Line at Wikipedia. Leave him alone. Blame me, block me. I have other accounts. to administrators Do not block Haber! Bock Localzuk and/or me. Localzuk is a worthless fucking tool who is g …


## 10 false negatives (missed labels: 4 overall + 3 threat + 3 identity_hate)

- **val row 9345** | true: toxic, threat, identity_hate | pred: clean | probs: toxic 0.7884, threat 0.0000, identity_hate 0.0506 | tokens: 112 | OOV: butterknife, daggers, dreamjerk, lubed
  > orders Dreamjerk has no right to order anyone to do anything. He is one of the more hated members of Wikipedia. Opponents just go on and do what you need to do. Daggers in back of Jack the Ripper, lubed by butter, falling on Rhobite. You wanna play, we can play. Butterknife. +++++++++++++++++******* …

- **val row 20082** | true: toxic, threat, identity_hate | pred: clean | probs: toxic 0.1090, threat 0.0000, identity_hate 0.0033 | tokens: 25 | OOV: none
  > Mmmm... yes... a pint of palestinian blood would be a fine dessert right now. - talk/email

- **val row 576** | true: threat | pred: clean | probs: threat 0.0002 | tokens: 22 | OOV: none
  > Please stop. If you continue to ignore our policies by introducing inappropriate pages to Wikipedia, you will be blocked.

- **val row 19456** | true: toxic, obscene, threat, insult | pred: toxic, insult | probs: toxic 0.9921, obscene 0.8901, threat 0.0003, insult 0.9417 | tokens: 386 (TRUNCATED) | OOV: 6500
  > "A RESPONSE TO . I ENTITLE THIS ""YOUR STUPIDITY"" The Assyrians are not extinct you incompetent bigoted fool. You lack all things historical you impotent maggot. The Assyrians have existed for over 6500 years you incumbent animal. The Assyrians first spoke Akkadian and in the 8th century the kings  …

- **val row 11901** | true: toxic, obscene, threat, insult | pred: toxic, severe_toxic, obscene, insult, identity_hate | probs: toxic 1.0000, severe_toxic 0.9889, obscene 1.0000, threat 0.0240, insult 0.9998, identity_hate 0.9969 | tokens: 973 (TRUNCATED) | OOV: wanta
  > {{unblock}}LIFETIME BAN LIFETIME BAN I WANTA FUCKIN LIFETIME BAN!!! BAIL OUT, BEAVIS AND BUTTHEAD, JESSICA SIMPSON, NIGGAZ, 420 MOONSHINE, FUCK!{{unblock}}LIFETIME BAN LIFETIME BAN I WANTA FUCKIN LIFETIME BAN!!! BAIL OUT, BEAVIS AND BUTTHEAD, JESSICA SIMPSON, NIGGAZ, 420 MOONSHINE, FUCK!{{unblock}}L …

- **val row 3569** | true: threat, insult | pred: toxic | probs: toxic 0.8610, threat 0.0519, insult 0.5092 | tokens: 35 | OOV: none
  > hey!!!!!!! thank you for re-editing my contributions it was really kind of you!!!!!!! i hope you get bowel cancer

- **val row 11252** | true: toxic, threat | pred: clean | probs: toxic 0.4064, threat 0.1469 | tokens: 7 | OOV: none
  > BAN ME AND DIE ^_^

- **val row 19044** | true: toxic, identity_hate | pred: clean | probs: toxic 0.0457, identity_hate 0.0048 | tokens: 8 | OOV: none
  > I HOPE ISRAEL GETS WIPED OFF THE MAP

- **val row 13018** | true: toxic, severe_toxic, obscene, insult, identity_hate | pred: toxic, obscene | probs: toxic 0.9977, severe_toxic 0.6239, obscene 0.9862, insult 0.8853, identity_hate 0.0088 | tokens: 64 | OOV: none
  > Fuck off you anti-bulgarian turkic whores!!! А на тебе джингибинги да ти еба путката майна лайно нещастно!!!

- **val row 12982** | true: toxic, identity_hate | pred: clean | probs: toxic 0.1529, identity_hate 0.0318 | tokens: 34 | OOV: none
  > "::You ought to get cool, it's not like I held up a mirror to you and called out ""white trash"". - 19 May 2015 "


## Findings

**Per-class errors.** toxic and obscene are balanced (precision and recall about 0.8). The rare classes over-flag: severe_toxic has 287 FP vs 158 TP (precision 0.355), and recall is higher than precision for every rare class. This matches the pos_weight class weights, which push the model towards predicting positives. The toxic vs severe_toxic boundary is also subjective.

### False positives: patterns
1. **Label noise.** Rows 16436 and 11996 are direct personal attacks with profanity but are labelled clean (10079 and 4681 are borderline). Here the model is arguably right, so part of the measured FP count is annotation noise.
2. **Keyword triggering without context.** The model fires on profanity or slurs that are quoted or used positively: rap lyrics (6153), a list of TV content ratings quoting the words (1367), "dance my ass off" in a happy comment (682), 23035. TextCNN sees 2-4 token windows and max-pooling keeps only the strongest window, so one strong word can decide the output; it cannot separate using a word from mentioning it.
3. **Identity terms.** Row 8819 mentions a religious group without attacking it and gets identity_hate 0.9995.
4. **Over-confidence.** All 10 FPs have probabilities >= 0.97, most >= 0.999, consistent with the rising validation loss.
5. **Length.** 8 of 10 FPs are longer than 128 tokens (vs 16.5% of all comments). Longer comments give more chances for a trigger word to appear (hypothesis, not tested).

### False negatives: patterns
1. **Implicit threats.** 20082 (violent metaphor about a group), 3569 (polite thanks, then wishing illness), 9345 (weapon imagery) contain no obvious threat keyword; the threat is in the meaning of the whole sentence. 11252 ("ban me and die") gets threat 0.147, far below its 0.9973 threshold: with such a high threshold only very explicit threats are caught.
2. **identity_hate without slurs.** 19044 (wishing a country to disappear) and 12982 (a quoted insult about a group). 13018 is half Bulgarian: the tokenizer keeps only a-z words, so the Cyrillic part becomes meaningless single characters.
3. **Truncation.** 19456 and 11901 are labelled threat but no threat appears in the start of the text; both are longer than 128 tokens, so the threat may be in the truncated part.
4. **Label inconsistency.** 576 is a standard Wikipedia warning ("you will be blocked") labelled threat. Quoted slurs are labelled identity_hate in 12982 but clean in the lyrics of 6153, so annotators are not consistent about quoted text, which limits any model.

## Did GloVe help rare classes?
Partly. In the ablation (same config and seed, random vs GloVe initialisation), GloVe raised macro-F1 from 0.6259 to 0.6486 and reached its best epoch at 3 instead of 9. The gain was large for identity_hate (0.461 -> 0.533) but almost zero for threat (0.558 -> 0.561). The error analysis suggests why: identity_hate often depends on which words appear (group and country names), and GloVe places related group names close together, so the model can generalise from groups seen in training to rarer ones. Threats depend on the meaning of the whole sentence (implicit threats, sarcasm, wishes of harm), which better word vectors alone cannot capture. This is a likely explanation, not a proven one: it is a single seed, and threat has only 71 validation positives, so one example changes its F1 by about 0.01.
