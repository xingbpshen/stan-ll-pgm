import re

def parse_response(message, targets):
    assert targets == ['stan_model']

    def _parse(_message, _target):
        if _target == 'stan_model':
            pattern = r"MODEL START(.*?)MODEL END"
            matches = re.findall(pattern, text, re.DOTALL)
            return [match.strip() for match in matches]
        else:
            pass