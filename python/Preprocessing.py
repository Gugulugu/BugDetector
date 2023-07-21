import re


class PreprocessingData:

    symbol_to_word_dict = {
        "!": "Exclamation",
        "@": "At",
        "#": "Hash",
        "$": "Dollar",
        "%": "Percent",
        "^": "Caret",
        "&": "Ampersand",
        "*": "Asterisk",
        "(": "OpenParenthesis",
        ")": "CloseParenthesis",
        "-": "Hyphen",
        "_": "Underscore",
        "+": "Plus",
        "=": "Equals",
        "{": "OpenCurlyBrace",
        "}": "CloseCurlyBrace",
        "[": "OpenSquareBracket",
        "]": "CloseSquareBracket",
        "|": "Pipe",
        "\\": "Backslash",
        "/": "Slash",
        ":": "Colon",
        ";": "Semicolon",
        "\"": "DoubleQuote",
        "'": "SingleQuote",
        "<": "LessThan",
        ">": "GreaterThan",
        ",": "Comma",
        ".": "Dot",
        "?": "QuestionMark",
        "~": "Tilde",
        "`": "Backtick",
        "<=": "LessThanOrEqualTo",
        ">=": "GreaterThanOrEqualTo",
        "==": "StrictEqual",
        "===": "AbstractEqual",
        "!=": "NotEqual",
        "&&": "And",
        "||": "Or",
        "++": "Increment",
        "--": "Decrement",
        "+=": "PlusEquals",
        "-=": "MinusEquals",
        "*=": "TimesEquals",
        "/=": "DivideEquals",
        "%=": "ModuloEquals",
        "<<": "LeftShift",
        ">>": "RightShift",


    }

    @staticmethod
    def preprocess_text(text):
        # Insert spaces before and after special characters
        text = re.sub('(ID:)', 'ID: ', text)
        text = re.sub('(LIT:)', 'LIT: ', text)
        text = re.sub('(na)', '', text)

        #results of logs/hparam_tuning_test_split/
        #text = re.sub('(:)', ' : ', text)
        # Insert space before a word that starts with a capital letter
        #text = re.sub('(?<=[a-z])(?=[A-Z])', ' ', text)
        # Insert space between a word and a number
        #text = re.sub('(?<=[a-zA-Z])(?=[0-9])', ' ', text)
        #text = re.sub('(?<=[0-9])(?=[a-zA-Z])', ' ', text)
        return text
    
    @staticmethod
    def symbols_to_text(text):
        for symbol, word in PreprocessingData.symbol_to_word_dict.items():
            # Use re.escape to escape special regex characters
            text = re.sub(re.escape(symbol), word, text)
        return text
