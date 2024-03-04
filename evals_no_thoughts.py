import json
import os
import re
from typing import Callable

from dotenv import load_dotenv
from openai import OpenAI
from parea.schemas.log import Log

from retry_manager import retry_decorator


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def eval_fun_factory(model: str, eval_fun_name: str, messages: list[dict[str, str]]) -> Callable[[Log], float]:
    @retry_decorator
    def eval_fun(log: Log) -> float:
        question = log.inputs['question']
        answer = str(log.output)

        replace_kwargs = {
            "question": question,
            "answer": answer,
        }

        def replace(match):
            var_name = match.group(1)
            return str(replace_kwargs.get(var_name, match.group(0)))

        formatted_messages = []
        for m in messages:
            formatted_messages.append({
                "role": m['role'],
                "content": re.sub(r"{{(\w+)}}", replace, m['content'])
            })

        completion = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=formatted_messages,
            response_format={"type": "json_object"}
        )
        final_verdict = json.loads(completion.choices[0].message.content).get('final_verdict', '').lower()
        return float(final_verdict == 'true')

    eval_fun.__name__ = eval_fun_name
    return eval_fun


gpt_35_turbo_0125_0_shot = eval_fun_factory(
    'gpt-3.5-turbo-0125',
    'gpt_35_turbo_0125_0_shot',
    [
        {
            "role": "user",
            "content": '''You are given a question and a list of answers. The answers were retrieved from a database which contains the question answer pairs. You need to decide if any of the given answers is the answer to the given question.

Question:
{{question}}

Answers:
{{answer}}

Answer in the following JSON format:
{"final_verdict": "<true|false>"}'''
        },
    ],
)
task_message = {
    "role": "user",
    "content": '''You are given a question and a list of answers. The answers were retrieved from a database which contains the question answer pairs. You need to decide if any of the given answers is the answer to the given question.

Answer in the following JSON format:
{"final_verdict": "<true|false>"}'''
}
final_message = {
    "role": "user",
    "content": '''Question:
{{question}}

Answers:
{{answer}}'''
}
false_few_shot_sample_1 = [
    {
        "role": "user",
        "content": '''Question:
What was the patient's principal diagnosis in the discharge summary and what chemotherapy was administered to treat it?

Answers:
[{'id': 21751, 'answer': 'The patient was diagnosed with sarcoma with asbestosis and received appropriate treatment for it, as stated in the discharge summary. However, the specific treatment used was not mentioned in the given information.'}, {'id': 21204, 'answer': "The patient's diagnosis, as stated in the given discharge summary, was Kaposi sarcoma. The cause of death was not stated in the report."}, {'id': 15507, 'answer': "According to the given discharge summary, the patient was started on HAART medication before receiving chemotherapy for Kaposi's sarcoma (KS) of the lung in concern of immune reconstitution syndrome. Further details on the specific drugs and dosages used for the treatment are not provided in the given information."}, {'id': 10550, 'answer': 'The diagnosis established for the patient in the given discharge summary was generalized lymphadenopathy. Antibiotics and chemotherapy were initiated as treatments.'}, {'id': 8080, 'answer': "The patient's diagnosis on discharge was acute pericarditis, and they were prescribed a regimen of prednisone and colchicine."}, {'id': 17890, 'answer': "The patient in the given discharge summary was diagnosed with extranodal NK/T-cell lymphoma-associated HPS and was treated with DDGPE chemotherapy regimen, supportive care including broad-spectrum antibiotics, antiviral therapy, plasma and platelet infusions, liver protectants, and sufficient fluid supply. This treatment plan led to significant improvements in the patient's symptoms and achieved complete remission after six cycles. However, the patient presented with chronic viral hepatitis with a marginally elevated HBV replication rate even after discharge."}, {'id': 32932, 'answer': "The patient's diagnosis was recurrent invasive extramammary Paget's disease, and they underwent multiple treatments including wide excision and reconstruction, IMRT, and palliative chemotherapy. The patient was discharged for further observation and treatment by their oncologist. No specific medications were mentioned in the discharge summary. However, further treatment may include additional chemotherapy, radiation therapy, or other palliative care options."}, {'id': 1392, 'answer': "The patient's final diagnosis was CD20-negative diffuse large B-cell lymphoma, and they were initiated on CHOP with methotrexate during odd cycles for CNS prophylaxis. The response to chemotherapy was monitored, and follow-up visits with her hematologist-oncologist were recommended."}, {'id': 28915, 'answer': "The patient's admitting diagnosis was new onset heart failure, while the discharge diagnosis was high-grade pleomorphic sarcoma with tumor recurrence and progression to woven bone. No medications were prescribed at the time of discharge, and no follow-up is required since the patient passed away. There are no discharge instructions provided."}, {'id': 12066, 'answer': "The patient's admitting diagnosis was Alzheimer's disease, and the principal diagnosis at discharge was Anti-NMDAR encephalitis."}, {'id': 19744, 'answer': "The patient's initial diagnosis was subacutely progressive pulmonary nodule with a final diagnosis of Mycobacterium avium complex (MAC) after further evaluation and treatment at the hospital. As for the healthcare professional's instruction, the patient was advised to rest well, avoid contact with individuals with contagious diseases, and follow up with his healthcare provider to monitor his recovery."}, {'id': 14837, 'answer': 'The final diagnosis of the patient mentioned in the discharge summary is paraganglioma of the broad ligament, and the treatment provided was complete excision of the lesion. Additionally, the patient was advised to follow a healthy diet and lifestyle and attend follow-up appointments as scheduled.'}, {'id': 10595, 'answer': "The patient's diagnosis was DDS with metastatic ovarian cancer to the brain and spinal cord, and her ultimate disposition was deceased. No follow-up recommendations were made due to the patient's passing."}, {'id': 8878, 'answer': "The patient's diagnosis upon discharge was central diabetes insipidus secondary to metastatic prostate carcinoma. The healthcare professional recommended the continuation of hormonal treatment for disease management and advised the patient to follow up with their primary care physician and oncologist for close disease monitoring purposes. The benefits of radiation therapy and chemotherapy were explained to the patient, but they refused both treatments."}, {'id': 29702, 'answer': 'According to the discharge summary, the 16-year-old female patient diagnosed with novel influenza A (H1N1) pneumonia was treated with oseltamivir phosphate, levofloxacin, and ceftriaxone.'}, {'id': 31590, 'answer': "According to the given discharge summary, the patient's diagnosis was paraneoplastic syndrome related to adenocarcinoma of the lung that secreted GM-CSF. The management plan involved continued monitoring for any worsening of symptoms, further evaluation for potential clinical trial participation, and outpatient follow-up care with routine oncology appointments. The patient was also educated on precautions to reduce the risk of infection, and the family was informed of the patient's diagnosis and treatment plan."}, {'id': 439, 'answer': 'The final diagnosis of the patient in the given discharge summary was a malignant mesothelioma, biphasic type. The patient refused further imaging and treatment, and was referred to palliative care for management of her symptoms. Therefore, no treatment plan was offered to the patient other than symptom management under the care of palliative care.'}, {'id': 20303, 'answer': 'According to the discharge summary, the patient was diagnosed with MSS colon cancer and received no prior treatment. However, no treatment plan was mentioned in the summary. Therefore, it is unanswerable as to what the treatment plan for the patient was.'}, {'id': 1358, 'answer': 'The final diagnosis for the patient presented in the discharge summary was an atypical pneumonia caused by Histoplasma capsulatum, and he exhibited an immune reconstitution inflammatory syndrome after receiving treatment with liposomal amphotericin B. He was started on liposomal amphotericin B and transitioned to oral itraconazole for the treatment.'}, {'id': 33263, 'answer': 'The final diagnosis for the patient was MPAL (T/Myeloid), and they received treatment with conventional karyotyping and monitoring for recurrence. The patient was advised to follow-up with their primary care physician and oncologist for regular check-ups and to avoid the risk factors of lymphoma. The discharge summary did not provide information on the specific medication used for treatment.'}]'''
    },
    {
        "role": "assistant",
        "content": '''{"final_verdict": "false"}'''
    },
]
false_few_shot_sample_2 = [
    {
        "role": "user",
        "content": '''Question:
In what way can the term 'xanthogranulomatous inflammatory process with histiocytic granulomas' be paraphrased to make it easier for a layman to understand in the context of the given discharge summary?

Answers:
[{'id': 19163, 'answer': 'The medical terminology used in the discharge summary related to the patient\'s eye condition can be paraphrased to make it more understandable for patients and their families. Instead of using complex terms such as "nodular scleritis inferotemporally," "granulomatosis with polyangiitis," and "orbital inflammatory pseudotumor," the healthcare team could use language that is easier to understand, such as "inflammation in the right eye" and "swelling around the eye." This will help the patient and their family better understand the condition and treatment. However, the discharge summary already included some simpler terms, such as "eye pain" and "redness," which are more easily understood by patients and their families.'}, {'id': 16467, 'answer': "Yes, I can simplify the medical terms used in the discharge summary for the patient and their family members. The patient was admitted to the hospital due to swelling and ulcers in the mouth for the past few years, with a diagnosis of Granulomatous cheilitis (GC). During the hospital stay, the patient went through various tests which indicated a soft tissue lesion, which potentially resulted in the swelling and ulcers. Laboratory tests ruled out any significant underlying medical condition such as tuberculosis, IBD, Crohn's disease and Helicobacter pylori. The patient was prescribed high-dose PPI therapy for chronic gastritis and high-dose infliximab, which helped him recover well. The discharge summary mentions that the patient was discharged in a stable condition with instructions to continue the prescribed medication and a follow-up appointment with the gastroenterology clinic."}, {'id': 18905, 'answer': 'Yes, the medical term "granulomatosis with polyangiitis" can be simplified as GPA, as indicated in the discharge summary.'}, {'id': 11358, 'answer': 'One way to simplify complex medical terminology in the discharge summary for patient comprehension is to replace medical jargon with layman\'s terms. For example, "muffled heart sounds" can be described as a "heart murmur" and "lymphocytic effusion" can be described as "fluid containing immune cells." Additionally, using visual aids such as diagrams or illustrations, and providing patient education materials can further enhance understanding. However, it should be noted that certain medical terms may be necessary for accuracy and cannot be completely substituted.'}, {'id': 9754, 'answer': 'The technical medical terms used in the diagnostic findings section of the discharge summary could be rephrased in simpler language to aid in patient comprehension by using layman\'s terms. For example, instead of using "large atypical cells with pleomorphic atypic nuclei, prominent nucleoli, and moderate oeosinophilic cytoplasm arranged in a lobular and solid pattern," the healthcare professional could say "abnormal cells that look different than normal cells and are arranged in a specific shape." However, it is important to note that some medical terms may be difficult to simplify without losing important information, so it is important to balance accuracy and patient comprehension.'}, {'id': 37767, 'answer': 'As per the discharge summary, the patient had a history of chronic bone infection with recurrent osteomyelitis and osteolytic lesions in the mandible. The patient had received treatment for a misdiagnosis of chronic granulomatous disease with IVIG. The patient should maintain good hygiene, follow a healthy lifestyle, and inform the healthcare team promptly in case of any fever, respiratory infections, or cellulitis. It is recommendable that the primary care physician be consulted for further management. However, due to the limited information provided, it might be challenging to rephrase medical terminology and treatments comprehensibly for the patient and their family.'}, {'id': 282, 'answer': 'One way to use paraphrasing in the discharge summary is by simplifying medical terms and using layman\'s terms to describe the patient\'s condition and treatment plan. This can make it easier for the patient and their family to understand the medical details and follow-up instructions. For example, instead of using technical terms like "en-bloc low anterior rectosigmoid resection," healthcare professionals can use simpler terms like "removal of the tumor and a part of the intestine."'}, {'id': 25136, 'answer': "Based on the histopathology report, the 71-year-old female patient had acute on chronic cholecystitis with xanthogranulomatous inflammation. This means that there was an ongoing inflammation of the gallbladder that had worsened over time and was accompanied by the accumulation of certain cells called xanthogranulomas. These findings are consistent with the patient's symptoms of pain abdomen and fever, as well as the thickening of the gallbladder wall detected on the USG."}, {'id': 36952, 'answer': 'To aid in better patient comprehension regarding their Leishmaniasis infection caused by L. tropica, the language used in the discharge summary can be simplified by defining medical terms such as "muco-cutaneous lesions" and "lymphadenopathy" in simpler language. Additionally, the use of acronyms like "DAT" can be explained in full. It would also be helpful to provide more information about the diagnosis and treatment options for the patient. However, the discharge summary does not provide information about the patient\'s treatment or follow-up plan, so this cannot be included in any simplified language.'}, {'id': 11583, 'answer': 'The task of paraphrasing can be applied to simplify the medical terminology used in the discharge summary by using simpler and more commonly understood language to explain the patient\'s condition, treatment, and outcome. For example, we can rephrase "acute pericarditis with moderate pericardial effusion" to "heart inflammation with moderate fluid buildup around the heart," or "prednisone dose was tapered down to 12.5 mg" to "the amount of prednisone taken was slowly reduced." This can help improve patient understanding and engagement in their own care. However, some medical terms and phrases may be necessary for accuracy and specificity, and some complexity may be unavoidable in certain cases.'}, {'id': 8079, 'answer': 'The medical term "dermal granulomatous infiltrate with associated eosinophils and a vaguely horizontally palisaded pattern with necrobiosis" in the diagnosis section of the discharge summary refers to a specific type of inflammation in the patient\'s skin that includes the infiltration of certain immune cells known as eosinophils, and a pattern of tissue damage called necrobiosis. It also includes the arrangement of the skin cells in a certain way that is described as a vaguely horizontally palisaded pattern.'}, {'id': 21145, 'answer': "Paraphrasing the discharge summary in layman's terms can help the patient understand their diagnosis and treatment plan better. It is crucial to use simple language and provide explanations for medical terms to ensure comprehension. Following up with the patient in the outpatient cardiology clinic will also help address any questions or concerns they may have."}, {'id': 34003, 'answer': 'To enhance patient comprehension, the medical terminology in the hospital course section of the discharge summary could be explained in simpler terms. This could include clarifying terms such as "cytoreductive surgery," "hyperthermic intraperitoneal chemotherapy," and "total peritonectomy." Additionally, the diagnosis could be explained in more understandable terms, such as using "cancer" instead of "malignant process of independent origins." However, it is important to note that some of the medical terminology used in the summary may be difficult to fully explain without sacrificing accuracy and completeness.'}, {'id': 14326, 'answer': "The discharge summary states that the patient suffered from myelodysplastic syndrome with excess blasts and acute myeloid leukemia, Clostridium difficile infection, pneumonia, superficial venous thrombosis, and Sweet's syndrome. Despite treatment, the patient's condition got worse and she eventually passed away. Unfortunately, the information provided in the discharge summary is too technical and complex to be effectively paraphrased for communication with the patient and their family. It would be best to consult with the treating physician to provide a simplified and understandable explanation of the patient's medical conditions and treatment to the patient and their family."}, {'id': 2739, 'answer': 'To improve communication with the patient and their family, medical jargon in the discharge summary can be reduced or paraphrased. For example, instead of using terms such as "pleomorphic spindle cells within a myxoid stroma containing thin-walled and compressed vessels," simpler language can be used such as "cells in the mass looked abnormal under a microscope." It is important to use clear and simple language to help patients and families understand their diagnoses and treatments. However, it is important to note that some medical terms may be unavoidable, and it is essential to provide explanations or definitions to ensure understanding.'}, {'id': 488, 'answer': 'To simplify the language of the patient\'s discharge summary for better understanding, medical jargon and complex terms can be replaced with simpler language. For example, instead of using terms like "loculated pleural effusion" or "polymorphonuclear predominance," we can use phrases like "fluid buildup in the lining of the lungs" and "increased white blood cells with a certain type of cell predominance." Additionally, instead of using the term "hydatid cyst," we can explain it as a type of infection caused by a parasite. However, some medical terminology may be necessary to accurately convey the patient\'s condition and treatment.'}, {'id': 19435, 'answer': 'To improve understanding for patients and their families, the discharge summary could be rephrased by using simpler language and explaining medical terms. Medical jargon like "heparin-induced thrombocytopenia" could be described as a condition where low blood platelets occur as a result of blood clots. Additionally, instead of describing the course of treatment in technical terms, it could be explained as a step-by-step process that the patient underwent to address the medical issue. This could help patients and their families more fully understand the situation and feel more involved in the treatment decisions.'}, {'id': 14718, 'answer': 'To simplify the language in the discharge summary for better patient communication, you could use terms like "persistent dry cough," "fluid buildup in the lungs," "radiation treatment," "a medication called ibrutinib," and "follow-up testing to check progress." Unfortunately, the summary does not provide enough details on the patient\'s acute coronary syndrome and its complications to simplify that part of the report.'}, {'id': 36299, 'answer': 'The use of paraphrasing techniques can aid in simplifying the medical jargon used in this discharge summary for the patient and their family to understand the diagnosis and treatment of idiopathic hematoma causing impending compartment syndrome by using simpler language and synonyms to convey the same information. For example, instead of using "impending compartment syndrome," one could say "swelling that could have caused damage." The medical details can be simplified using laymen terms such as "surgery" instead of "fasciotomy and decompression." It is important to use language that the patient and family can comprehend to ensure they understand the diagnosis and treatment plan.'}, {'id': 3047, 'answer': 'To make the discharge summary more easily understood for non-medical professionals, the provider can simplify the medical terms used. For example, systemic lupus erythematosus (SLE) can be described as an autoimmune disease that causes the immune system to attack healthy tissues, and Laparoscopic Sleeve Gastrectomy (LSG) can be explained as a weight-loss surgery that removes a portion of the stomach. Additionally, the provider can provide more context and explanation for certain terms such as perioperative steroid cover, HbA1c, and Mazindol, to ensure the patient fully understands their treatment and care. As for the instruction provided, it appears to be fully answerable within the given discharge summary.'}]'''
    },
    {
        "role": "assistant",
        "content": '''{"final_verdict": "false"}'''
    },
]
gpt_35_turbo_0125_1_shot_false_sample_1 = eval_fun_factory(
    'gpt-3.5-turbo-0125',
    'gpt_35_turbo_0125_1_shot_false_sample_1',
    [
        task_message,
        *false_few_shot_sample_1,
        final_message
    ],
)
gpt_35_turbo_0125_1_shot_false_sample_2 = eval_fun_factory(
    'gpt-3.5-turbo-0125',
    'gpt_35_turbo_0125_1_shot_false_sample_2',
    [
        task_message,
        *false_few_shot_sample_2,
        final_message
    ],
)
gpt_35_turbo_0125_2_shot_false_1_false_2 = eval_fun_factory(
    'gpt-3.5-turbo-0125',
    'gpt_35_turbo_0125_2_shot_false_1_false_2',
    [
        task_message,
        *false_few_shot_sample_1,
        *false_few_shot_sample_2,
        final_message
    ],
)
gpt_35_turbo_0125_2_shot_false_2_false_1 = eval_fun_factory(
    'gpt-3.5-turbo-0125',
    'gpt_35_turbo_0125_2_shot_false_2_false_1',
    [
        task_message,
        *false_few_shot_sample_2,
        *false_few_shot_sample_1,
        final_message
    ],
)


def hit_rate_top_factory(limit: int = 10) -> Callable[[Log], float]:
    def hit_rate_top(log: Log) -> float:
        question_id = log.inputs['row_id']
        output = json.loads(log.output)
        answer_ids = [answer['id'] for answer in output][:limit]
        return float(question_id in answer_ids)

    hit_rate_top.__name__ = f'hit_rate_top_{limit}'
    return hit_rate_top


hit_rate_top_5 = hit_rate_top_factory(5)
hit_rate_top_10 = hit_rate_top_factory(10)
hit_rate_top_20 = hit_rate_top_factory(20)


def mrr_top_factory(limit: int = 10) -> Callable[[Log], float]:
    def mrr_top(log: Log) -> float:
        question_id = log.inputs['row_id']
        output = json.loads(log.output)
        answer_ids = [answer['id'] for answer in output][:limit]
        if question_id in answer_ids:
            return 1 / (answer_ids.index(question_id) + 1)
        else:
            return 0

    mrr_top.__name__ = f'mrr_top_{limit}'
    return mrr_top


mrr_top_5 = mrr_top_factory(5)
mrr_top_10 = mrr_top_factory(10)
mrr_top_20 = mrr_top_factory(20)
