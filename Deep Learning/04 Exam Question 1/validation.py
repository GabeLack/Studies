import re
import numpy as np

class Validation:
    @staticmethod
    def validate_str_alpha(value:str):
        return isinstance(value,str) and value.strip().isalpha()

    @staticmethod
    def validate_str_alphanum(value:str):
        if " " in value:
            value.replace(" ","").isalpha()
        return isinstance(value,str) and value.strip().isalnum()
    
    @staticmethod
    def validate_country_code(value: str):
        return isinstance(value, str) and \
        value.startswith('+') and value[1:].isdigit()
        
    @staticmethod
    def validate_str_num(value:str):
        return isinstance(value,str) and value.strip().isnumeric()
    
    @staticmethod
    def validate_float(value):
        return isinstance(value,float)

    @staticmethod
    def validate_int(value):
        return isinstance(value,int)

    @staticmethod
    def validate_str(value):
        return isinstance(value,str)
    
    @staticmethod
    def validate_email(value:str):
        pattern = re.compile(r"^[a-zA-z0-9][A-Za-z0-9\._-]*@\w*\.\w{2,3}")
        return bool(re.match(pattern,value))

    @staticmethod
    def validate_password(value: str) -> bool:
        """Validate password. Password must have a minimum length of 8 characters and contain
        at least one number and one special character."""
        pattern = re.compile(r"^(?=.*[0-9])(?=.*[\\.!@$%^&*#-])[A-Za-z0-9!@#$%^&*\\.-]{8,}$")
        return bool(re.match(pattern, value))

    @staticmethod
    def validate_id_name(value:str):
        """Validate names with letter A-Ö both capital letters and lowercase
        but no space words have to be separated with _ """
        pattern = re.compile(r"([A-Öa-ö0-9_]+)")
        return bool(re.match(pattern,value))

    @staticmethod
    def validate_card_number(value:str):
        my_pattern = re.compile(r"\d{16}")
        return re.search(pattern=my_pattern,string=value) is not None

    @staticmethod
    def validate_name(value:str):
        pattern = re.compile(r"[A-Öa-ö]?[A-Öa-ö]{2,}\s[A-Öa-ö]{2,}")
        return bool(re.match(pattern,value))

    @staticmethod
    def validate_vat_id(value:str):
        pattern = re.compile(r"[A-Z]{2,2}\s\d{8,12}")
        return bool(re.match(pattern,value))

    @staticmethod
    def validate_value_in_values(value,values):
        value_is_ok = False
        for val in values:
            if isinstance(value,type(val)):
                value_is_ok = True
            else:
                value_is_ok = False
                break
            if value_is_ok and value in values:
                return value_is_ok
            else:
                value_is_ok = False
                return value_is_ok

    @staticmethod
    def read_in_str_value(validation_function,message:str):
        while True:
            user_input = input(message)
            if validation_function(user_input):
                return user_input

    @staticmethod
    def read_in_int_value(validation_function,message:str):
        while True:
            user_input = input(message)
            if user_input.isnumeric() and validation_function(int(user_input)):
                return int(user_input)

    @staticmethod
    def read_in_float_value(validation_function,message:str):
        while True:
            try:
                user_input = input(message)
                if validation_function(float(user_input)):
                    return float(user_input)
            except:
                continue

    @staticmethod
    def validate_path(value: str) -> bool:
        """Validate path. Path must not contain whitespaces or special characters."""
        pattern = re.compile(r"^[^\s!@#$%^&*()_+{}\[\]:;<>,.?/~\\]+$")
        return bool(re.match(pattern, value))

    @staticmethod
    def validate_pos_ints(values):
        if isinstance(values, np.ndarray):
            values = values.tolist()
        return all(Validation.validate_int(value) and value > 0 for value in values)

    @staticmethod
    def validate_pos_ints_floats(values):
        if isinstance(values, np.ndarray):
            values = values.tolist()
        return all((Validation.validate_int(value) or Validation.validate_float(value)) \
            and value > 0 for value in values)

    @staticmethod
    def validate_pos_ints_floats_w_0(values):
        if isinstance(values, np.ndarray):
            values = values.tolist()
        return all((Validation.validate_int(value) or Validation.validate_float(value)) \
            and value >= 0 for value in values)

    @staticmethod
    def validate_tuple_length(values, length):
        if isinstance(values, tuple) and len(values)==length:
            return all((isinstance(value, int) or isinstance(value,float)) \
                and value > 0 for value in values)

    @staticmethod
    def validate_tuple_ints_floats(values):
        if isinstance(values, tuple):
            values = list(values)
        return all((isinstance(value, int) or isinstance(value, float)) for value in values)

    @staticmethod
    def validate_array_length(values, expected_length):
        if isinstance(values, np.ndarray):
            if len(values.shape) == 1:
                return len(values) == expected_length
            else:
                return all(len(row) == expected_length for row in values)