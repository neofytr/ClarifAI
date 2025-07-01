import openai

OPENAI_API_KEY = "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"

openai.api_key = OPENAI_API_KEY

try:
	response = openai.ChatCompletion.create(
		model="gpt-3.5-turbo",
		messages=[
			{"role": "user", "content": "Hello, GPT! What's 2 + 2?"}
		]
	)
	print(response['choices'][0]['message']['content'])

except openai.error.OpenAIError as e:
	print("OpenAI API Error:", e)
