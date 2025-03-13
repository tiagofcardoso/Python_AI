import openai

def test_openai_api_key(api_key):
    openai.api_key = api_key
    try:
        response = openai.Completion.create(
            engine="davinci",
            prompt="This is a test.",
            max_tokens=5
        )
        return response
    except Exception as e:
        return str(e)

api_key = "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"
response = test_openai_api_key(api_key)
print(response)
