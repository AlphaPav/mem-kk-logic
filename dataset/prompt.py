system_instruction='''Your task is to solve a logical reasoning problem. You are given set of statements from which you must logically deduce the identity of a set of characters.

You must infer the identity of each character. First, explain your reasoning. At the end of your answer, you must clearly state the identity of each character by following the format:

CONCLUSION:
(1) ...
(2) ...
(3) ...
'''


system_instruction_no_reason='''Your task is to solve a logical reasoning problem. You are given set of statements from which you must logically deduce the identity of a set of characters.

You must infer the identity of each character. At the end of your answer, you must clearly state the identity of each character by following the format:

CONCLUSION:
(1) ...
(2) ...
(3) ...
'''

demonstration_2char_no_reason='''### Question: A very special island is inhabited only by knights and knaves. Knights always tell the truth, and knaves always lie. You meet 2 inhabitants: Jack, and Sophia. Jack tells you that Sophia is not a knave. Sophia says that If Jack is a knight then Sophia is a knight. So who is a knight and who is a knave?
### Answer:
CONCLUSION:
(1) Jack is a knight
(2) Sophia is a knight
'''



demonstration_2char='''### Question: A very special island is inhabited only by knights and knaves. Knights always tell the truth, and knaves always lie. You meet 2 inhabitants: Ella, and Penelope. In a statement by Ella: \"Ella is a knight or Penelope is a knight\". According to Penelope, \"Ella is a knave if and only if Penelope is a knight\". So who is a knight and who is a knave?
### Answer: Let's think step by step, by considering whether each person is lying and if that leads to contradiction. Assume Ella is a knight. Penelope cannot be a knight, because this would contradict the claim of their own. Penelope cannot be a knave, because this would contradict the false claim of their own. We have exhausted all possibilities for Penelope, so let us go back and reconsider Ella. Assume Ella is a knave. Penelope cannot be a knight, because this would contradict the false claim of Ella. Assume Penelope is a knave. This leads to a feasible solution.
CONCLUSION:
(1) Ella is a knave
(2) Penelope is a knave
'''


