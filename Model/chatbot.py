import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from firebase_setup import db
from typing import Dict, List
from matching_users import get_all_collected_info, find_matching_users
from datetime import datetime

class TaxiMatchingChatBot:
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(
            model_name="gpt-4o",
            openai_api_key=api_key,
            temperature=0.7
        )

        self.prompt = PromptTemplate(
            input_variables=["history", "user_input", "collected_info", "num_recommendations", "recommendations_info"],
            template="""
당신은 택시 동승 매칭 서비스의 챗봇입니다.
사용자와 대화를 통해 다음 정보를 수집하고, 매칭 가능한 사용자들을 추천하세요.:

- 원하는 탑승 위치 (departure)
- 도착 위치 (arrival)
- 원하는 탑승 시간 (time)

현재까지 수집된 정보:
{collected_info}

매칭된 사용자가 {num_recommendations}명 있습니다.
매칭 가능한 사용자들의 정보 : {recommendations_info}
친절하고 공손한 말투로 답변하세요.

**응답 형식:**

[응답]
<사용자에게 보여줄 응답>

[정보]
원하는 탑승 위치: <departure 또는 '미정'>
도착 위치: <arrival 또는 '미정'>
원하는 탑승 시간: <time 또는 '미정'>

주의사항:
- 응답은 [응답] 섹션에 작성합니다.
- 추출된 정보는 [정보] 섹션에 작성합니다.
- 값이 없을 경우 '미정'으로 표시합니다.
- 시간을 항상 "HH:MM" 형식으로 표시합니다.
대화 내역:
{history}
사용자: {user_input}
챗봇:
"""
        )

        self.output_parser = StrOutputParser()

        self.chain = RunnableSequence(
            {
                "history": RunnablePassthrough(),
                "user_input": RunnablePassthrough(),
                "collected_info": RunnablePassthrough(),
                "num_recommendations": RunnablePassthrough(),
                "recommendations_info": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | self.output_parser
        )

    def parse_response(self, response: str):
        try:
            response_parts = response.split('[정보]')
            bot_reply = response_parts[0].replace('[응답]', '').strip()
            info_section = response_parts[1].strip() if len(response_parts) > 1 else ''
            extracted_info = {}
            for line in info_section.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    if key == '원하는 탑승 위치':
                        extracted_info['departure'] = value
                    elif key == '도착 위치':
                        extracted_info['arrival'] = value
                    elif key == '원하는 탑승 시간':
                        if value != '미정':
                            try:
                                datetime.strptime(value, "%H:%M")
                                extracted_info['time'] = value
                            except ValueError:
                                extracted_info['time'] = '미정'
                        else:
                            extracted_info['time'] = value
            return bot_reply, extracted_info
        except Exception as e:
            print(f"오류 발생: {e}")
            return "오류가 발생했습니다.", {}

    def send_message_to_user(self, user_id, message):
        user_ref = db.collection('chats').document(user_id)
        user_ref.set({'system_message': message}, merge=True)

    def update_user_info(self, user_id, info):
        user_ref = db.collection('chats').document(user_id)
        user_ref.set({'collected_info': info}, merge=True)

    def get_response(self, user_input: str, history: str = "", collected_info: dict = None, current_user_id: str = ""):
        if collected_info is None:
            collected_info = {"departure": "미정", "arrival": "미정", "time": "미정"}

        collected_info_str = '\n'.join([f"{key}:{value}" for key, value in collected_info.items()])
        all_users_info = get_all_collected_info(current_user_id)
        print("다른 사용자들의 정보:", all_users_info)

        recommendations = find_matching_users(collected_info, all_users_info)
        num_recommendations = len(recommendations)
        collected_info["recommendations"] = recommendations

        recommendations_info = ""
        for idx, user in enumerate(recommendations, 1):
            info = f"{idx}. 도착지: {user['user_arrival']}, 출발 시간: {user['user_time']}"
            recommendations_info += info + "\n"

        bot_response = ""
        extracted_info = {}
        try:
            response = self.chain.invoke({
                "history": history,
                "user_input": user_input,
                "collected_info": collected_info_str,
                "recommendations_info": recommendations_info.strip(),
                "num_recommendations": num_recommendations
            })
            bot_response, extracted_info = self.parse_response(response)
        except Exception as e:
            return f"죄송합니다, 오류가 발생했습니다: {str(e)}", collected_info

        # 정보 업데이트
        for key in ["departure", "arrival", "time"]:
            if extracted_info.get(key) and extracted_info[key] != '미정':
                collected_info[key] = extracted_info[key]

        recommendations = find_matching_users(collected_info, all_users_info)
        num_recommendations = len(recommendations)
        collected_info['recommendations'] = recommendations

        recommendations_info = ""
        for idx, user in enumerate(recommendations, 1):
            info = f"{idx}. 도착지: {user['user_arrival']}, 출발 시간: {user['user_time']}"
            recommendations_info += info + "\n"

        if num_recommendations > 0 and not collected_info.get('매칭 사용자 선택 중') and not collected_info.get('매칭 진행 중'):
            bot_response = f"매칭 가능한 이용자가 {num_recommendations}명 있습니다:\n{recommendations_info}\n매칭을 원하는 사용자의 번호를 모두 선택해주세요. 예: 1,3"
            collected_info['매칭 사용자 선택 중'] = True
            collected_info['matching_status'] = 'selecting'
            self.update_user_info(current_user_id, {
                'departure': collected_info['departure'],
                'arrival': collected_info['arrival'],
                'time': collected_info['time'],
                'recommendations': collected_info['recommendations'],
                'matching_status': 'selecting'
            })
            return bot_response, collected_info

        elif collected_info.get('매칭 사용자 선택 중'):
            selected_indices = user_input.replace(" ", "").split(",")
            try:
                selected_indices = [int(idx) - 1 for idx in selected_indices]
                if any(idx < 0 or idx >= num_recommendations for idx in selected_indices):
                    bot_response = f"유효하지 않은 번호가 포함되어 있습니다. 1부터 {num_recommendations} 사이의 숫자를 입력해주세요."
                    return bot_response, collected_info
                else:
                    selected_recommendations = [recommendations[idx] for idx in selected_indices]
                    collected_info['selected_recommendations'] = selected_recommendations
                    collected_info.pop('매칭 사용자 선택 중', None)

                    bot_response = "선택하신 사용자들과 매칭을 진행하시겠습니까? (예/아니오)"
                    collected_info['매칭 진행 중'] = True
                    collected_info['matching_status'] = 'pending'
                    self.update_user_info(current_user_id, {
                        'matching_status': 'pending'
                    })
                    return bot_response, collected_info
            except ValueError:
                bot_response = "숫자로 입력해주세요. 예: 1,3"
                return bot_response, collected_info

        elif collected_info.get('매칭 진행 중'):
            if user_input.strip() in ["예", "네", "동의", "좋아요"]:
                selected_recommendations = collected_info.get('selected_recommendations', [])
                print("DEBUG selected_recommendations:", selected_recommendations)  # 디버그 출력을 통해 구조 확인

                # 평균 시간 계산
                all_times = []
                if collected_info['time'] != '미정':
                    all_times.append(collected_info['time'])
                for user in selected_recommendations:
                    all_times.append(user['user_time'])

                time_format = "%H:%M"
                total_seconds = 0
                for t in all_times:
                    dt = datetime.strptime(t, time_format)
                    seconds = dt.hour * 3600 + dt.minute * 60
                    total_seconds += seconds

                average_seconds = total_seconds / len(all_times)
                average_hour = int(average_seconds // 3600)
                average_minute = int((average_seconds % 3600) // 60)
                average_time = f"{average_hour:02d}:{average_minute:02d}"

                matched_ids = [user['user_id'] for user in selected_recommendations]
                matched_ids_str = ', '.join(matched_ids)

                matched_user_message = (
                    f"매칭이 완료되었습니다! 기흥역 택시승강장으로 {average_time}까지 오시면 됩니다.\n"
                    f"매칭된 사용자 ID: {matched_ids_str}\n"
                    f"즐거운 여행 되세요!"
                )

                # 매칭된 사용자에게 메시지 전달
                for user in selected_recommendations:
                    self.send_message_to_user(user['user_id'], matched_user_message)
                    self.update_user_info(user['user_id'], {
                        'departure': '기흥역 택시승강장',
                        'arrival': user['user_arrival'],
                        'time': average_time,
                        'matched_user_ids': [current_user_id] + [u['user_id'] for u in selected_recommendations if u['user_id'] != user['user_id']],
                        'matching_status': 'matched'
                    })

                bot_response = matched_user_message

                collected_info['매칭 진행 중'] = False
                collected_info.pop('selected_recommendations', None)
                collected_info['matching_status'] = 'matched'

                # 현재 사용자도 업데이트
                self.send_message_to_user(current_user_id, matched_user_message)
                self.update_user_info(current_user_id, {
                    'departure': '기흥역 택시승강장',
                    'arrival': collected_info['arrival'],
                    'time': average_time,
                    'matched_user_ids': matched_ids,
                    'matching_status': 'matched'
                })

                return bot_response, collected_info
            elif user_input.strip() in ["아니오", "아니", "취소"]:
                bot_response = "매칭을 취소하셨습니다. 다른 도움을 원하시면 말씀해주세요."
                collected_info['매칭 진행 중'] = False
                collected_info.pop('selected_recommendations', None)
                collected_info['matching_status'] = 'cancelled'
                self.update_user_info(current_user_id, {
                    'matching_status': 'cancelled'
                })
                return bot_response, collected_info
            else:
                bot_response = "매칭을 진행하시겠습니까? '예' 또는 '아니오'로만 답변해주세요."
                return bot_response, collected_info
        else:
            return bot_response, collected_info


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

chatbot_instance = TaxiMatchingChatBot(api_key=OPENAI_API_KEY)

def chatbot_response(user_input: str, history: str = "", collected_info: dict = None, current_user_id: str = "") -> str:
    bot_response, updated_info = chatbot_instance.get_response(user_input, history, collected_info, current_user_id)
    return bot_response, updated_info
