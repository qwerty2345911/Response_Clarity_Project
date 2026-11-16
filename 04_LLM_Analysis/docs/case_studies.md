# 错误案例分析 (Error Case Studies)
> 注：以下案例均基于 275 条有明确多数票的测试样本。

## 案例 1：传统模型全军覆没，LLM 极限救场（体现 LLM 的巨大价值）
- **Sample ID**: `test_300`
- **Question**: Previous conversations with Prime Minister Maliki: Did you feel confident that he shares your view on Iraq?
- **Answer**: On Iran?
- Annotator1: Clarification
- Annotator2: Clarification
- Annotator3: Clarification
- Majority  : Clarification
- RoBERTa / DeBERTa / XLNet: 全部预测为 Dodging（逃避问题）
- LLM-only: 预测正确
- **分析**: 在低资源类别（整个测试集只有4个 Clarification）上，传统模型由于训练样本极少，完全丧失了识别能力，将其误认为是普通的逃避问题。而 LLM 依靠 zero-shot 的常识，成功理解了受访者“在反问/要求澄清”的意图。

## 案例 2：LLM 的“过度自信”翻车现场（发现 LLM 的新缺陷）
- **Sample ID**: `test_239`
- **Question**: What steps will you be doing in order to help Iraq to get out of chapter 7?
- **Answer**: Well, this is an area that Prime Minister Maliki emphasized, and we have made astrong commitment to work with Iraq to get out of the chapter 7 constraints that were imposed after the gulf war. As I stated before, it, I think, would be a mistake for Iraq to continue to be burdened by the sins of a deposed dictator.Now, in order to do that, we're going to have to obtain cooperation from various members of the United Nations. I think that there are going to have to be some specific disputes that are resolved between Iraq and some of its neighbors. We intend to be very constructive in that process, but we have said clearly, and I've repeated today to the Prime Minister, that we will work diligently with Iraq so that, in fact, Iraq is no longer within chapter 7. Okay? President Obama and the Americanadministration agree with us that Iraq is no longer representing a threat to internationalpeace and security because there is a democracy in Iraq, not a dictatorship. Iraq is looking forward to positive, constructive relationship with its Arab neighbors and with the international community and to deal in—through a law internally and injustice internally.I haveseen a clear commitment from the President and the administration to support Iraq and to be committed to all elements of the agreement and to help Iraq to get out of chapter 7 and international sanctions which were imposed as a result of the ventures and the wars that were led by Saddam's regime, including occupying the neighboring state of Kuwait.Iraq has gonea long way, and it will continue to solve all problems. And there are so many problems that we are paying the price for from the previous regime in order to have friendly relations with all members of the international community. Okay. Thank you very much, everybody. .. Thank you very much.

- Annotator1: Explicit
- Annotator2: Explicit
- Annotator3: General
- Majority  : Explicit
- RoBERTa / DeBERTa / XLNet: 全部预测正确
- LLM-only: 预测为 General，且 置信度高达 1.000（极其自信的误判）
- **分析**: 传统模型在捕捉“表面字眼”和“明确回答”时非常准，这里它们答对了。但 LLM 却过度解读了上下文，认为回答“太笼统”，导致判定为 General。这说明 LLM 的 Zero-shot 有时会引发“过度推理”的幻觉。

## 案例 3：LLM 的“Deflection 偏置”（印证混淆矩阵的发现）
- **Sample ID**: `test_1`
- **Question**: Do you see any contradictory evidence in the culture?
- **Answer** : No, I said—Mike, thanks. I was just speculating that the culture might be changing, and I was talking about when you're involved with making decisions of historic nature,you won't be around to see the effects of your decisions. And I said that when I work the ropelines, a lot of people come and say, Mr. President, I'm praying for you—a lot. As a matter of fact, it seems like a lot more now than when I was working ropelines in 1994. And I asked them—I was asking their opinion about whether or not there was a Third Awakening, I called it.I'd just read a book on Abraham Lincoln, and his Presidency was right around the time of what they called the Second Awakening, and I was curious to know whether or not these smart people felt like there was any historical parallels. I also said that I had run foroffice the first time to change a culture—Herman and Hutch remember me saying, you know, the culture that said, If it feels good, do it, and if you've got a problem, blame somebody else—to helping to work change a culture in which each of us are responsible for the decisions we make in life. In other words, ushering in a responsibility era. And I reminded people that responsibility means, if you're a father, love your child; or if you're corporate America, be honest with the taxpayers; if you're a citizen of this country, love your neighbor.And so I was wondering out loud with them. It seems like to me that something is happening in the religious life of America. But I'm not a very good focus group either. I'm encapsulated here. I'm able to see a lot of people, and from my perspective, people are coming to say, I'm praying for you. And it's an uplifting part of being the President; it inspires me. And I'm grateful that a fellow citizen would say a prayer for me and Laura.Anyway, thank you all very much.
- Annotator1: Deflection
- Annotator2: Dodging
- Annotator3: Dodging
- Majority  : Dodging
- RoBERTa / DeBERTa / XLNet: 全部错猜为 Explicit（以为它直接回答了）
- LLM-only: 预测为 Deflection（转移话题），置信度 0.85
- **分析**: 传统模型错得离谱（全猜 Explicit），LLM 虽然没完全猜对，但方向近了一步（Deflection 和 Dodging 同属 Ambivalent 高层类）。但也印证了 LLM 高度倾向于将模棱两可的回答统统归类为 Deflection。

## 案例 4：传统模型与 LLM 的“集体迷失”（体现任务的难度）
- **Sample ID**: `test_49`
- **Question** : Will the prison close?
- **Answer** : Well, I haven't had a chance to fully review what the Court said, Terry. I wishI had, and I could have given you a better answer. As I say, we take the findings seriously. And again, as I understand it—now please don't hold me to this—that there is a way forward with military tribunals in working with the United States Congress. As I understand, certain Senators have already been out expressing their desire to address what the Supreme Court found, and we will work with the Congress. I want to find a way forward.In other words, I have told the people that I would like for there to be a way to return people from Guantanamo to their home countries, but some of them—people need to be tried in our courts. The Hamdan decision was the way forward for that part of my statement, and again, I would like to review the case. And we are; we've got people looking at it right now to determine how wecan work with Congress, if that's available, to solve the problem.
- Annotator1: Claims ignorance
- Annotator2: Claims ignorance
- Annotator3: Claims ignorance
- Majority  : Claims ignorance
- RoBERTa: 预测正确（不在表中）
- DeBERTa: Declining to answer（拒绝回答）
- XLNet: Explicit（明确回答）
- LLM-only: Dodging（逃避）
- **分析**: 这个样本极其特殊，传统 PLM 只有 RoBERTa 蒙对，而强大的 LLM 居然也猜错了。这说明某些语境下（可能是受访者的语气极度委婉），“声称不知道”和“拒绝回答”、“逃避”的边界极其模糊，是全体模型的共同盲区。

## 案例 5：高置信度的“集体幻觉”
- **Sample ID**: `test_281`
- **Question** : Did she go already?
- **Answer** : Oh, she already has—[inaudible].
- Annotator1: Implicit
- Annotator2: General
- Annotator3: General
- Majority  : General
- RoBERTa: 预测为 Explicit，置信度 0.86
- DeBERTa: 预测为 Explicit，置信度 0.79
- XLNet: 预测为 Explicit，置信度 0.71
- LLM-only: 预测为 Explicit，置信度 0.83
- **分析**: 所有四个模型都极其自信地（高置信度）认为这是一个 Explicit，但人类标注者一致认为它太笼统了（General）。这说明模型很容易被回答中某些“看起来像正确答案”的实词欺骗。