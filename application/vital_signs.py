import re

class VitalSigns:
    def __init__(self):
        self.temperature = None  # Temperature
        self.blood_pressure = None  # Blood pressure
        self.heart_rate = None  # Heart rate
        self.respiration_rate = None  # Respiration rate
        self.oxygen_saturation = None  # Oxygen saturation
        
    def __str__(self):
        return (
            'Temperature: ' + str(self.temperature) + '\n' +
            'Blood Pressure: ' + str(self.blood_pressure) + '\n' +
            'Heart Rate: ' + str(self.heart_rate) + '\n' +
            'Respiration Rate: ' + str(self.respiration_rate) + '\n' +
            'Oxygen Saturation: ' + str(self.oxygen_saturation) + '\n'
        )

    
    def __repr__(self):
        return (
            'Temperature: ' + str(self.temperature) + '\n' +
            'Blood Pressure: ' + str(self.blood_pressure) + '\n' +
            'Heart Rate: ' + str(self.heart_rate) + '\n' +
            'Respiration Rate: ' + str(self.respiration_rate) + '\n' +
            'Oxygen Saturation: ' + str(self.oxygen_saturation) + '\n'
        )
    
def extract_numbers(s):
    num = 0
    for c in s:
        if '0' <= c <= '9':
            num = num * 10 + int(c)
        else:
            return str(num)
    return ""

def extract_vital_signs(row):
    global total_valid_lines
    elements = re.split(r'[ ,]+', row)
    vital_signs = VitalSigns()
    pos = 0
    
    while pos < len(elements):
        if re.match(r'^([Tt][.:]?$)|(Temp[.:]?$)', elements[pos]) and vital_signs.temperature is None:
            pos += 1
            vital_signs.temperature = elements[pos]
            pos += 1
            continue

        if re.match(r'^[Tt][0-9]+.*', elements[pos]) and vital_signs.temperature is None:
            vital_signs.temperature = elements[pos][1:]
            pos += 1
            continue
        
        if re.match(r'^[Tt][.:=][0-9]+.*', elements[pos]) and vital_signs.temperature is None:
            vital_signs.temperature = elements[pos][2:]
            pos += 1
            continue
        
        if vital_signs.temperature is None and re.match(r'^[0-9]+\.[0-9]$', elements[pos]):
            vital_signs.temperature = elements[pos]
            pos += 1
            continue

        if vital_signs.blood_pressure is None and re.match(r'^[Bb][Pp][0-9]+/[0-9]+$', elements[pos]):
            vital_signs.blood_pressure = elements[pos][2:]
            pos += 1
            continue

        if vital_signs.blood_pressure is None and re.match(r'^[Bb][Pp][.:=][0-9]+/[0-9]+$', elements[pos]):
            vital_signs.blood_pressure = elements[pos][3:]
            pos += 1
            continue

        if re.match(r'^[Bb][Pp][.:]?$', elements[pos]) and vital_signs.blood_pressure is None:
            pos += 1
            vital_signs.blood_pressure = elements[pos]
            pos += 1
            continue

        if vital_signs.heart_rate is None and re.match(r'^[Hh][Rr][0-9]+$', elements[pos]):
            vital_signs.heart_rate = elements[pos][2:]
            pos += 1
            continue

        if vital_signs.heart_rate is None and re.match(r'^[Hh][Rr][.:=][0-9]+$', elements[pos]):
            vital_signs.heart_rate = elements[pos][3:]
            pos += 1
            continue
        
        if vital_signs.heart_rate is None and re.match(r'^[Hh][Rr][0-9]+-[0-9]+$', elements[pos]):
            vital_signs.heart_rate = elements[pos][2:]
            pos += 1
            continue
        
        if vital_signs.heart_rate is None and re.match(r'^[Hh][Rr][.:]?$', elements[pos]) and vital_signs.heart_rate is None:
            pos += 1
            vital_signs.heart_rate = extract_numbers(elements[pos])
            pos += 1
            continue
        
        if re.match(r'^[Rr][Rr][.:]?$', elements[pos]) and vital_signs.respiration_rate is None:
            pos += 1
            vital_signs.respiration_rate = elements[pos]
            pos += 1
            continue
        
        if vital_signs.respiration_rate is None and re.match(r'^[Rr][Rr][0-9]+$', elements[pos]):
            vital_signs.respiration_rate = elements[pos][2:]
            pos += 1
            continue

        if vital_signs.respiration_rate is None and re.match(r'^[Rr][Rr][:=.][0-9]+$', elements[pos]):
            vital_signs.respiration_rate = elements[pos][3:]
            pos += 1
            continue

        if re.match(r'^[Hh][Rr][.:]?$', elements[pos]) and vital_signs.heart_rate is None:
            pos += 1
            vital_signs.heart_rate = extract_numbers(elements[pos])
            pos += 1
            continue
        
        if re.match(r'^[oO]2[.:]?$', elements[pos]) and vital_signs.oxygen_saturation is None:
            pos += 1
            vital_signs.oxygen_saturation = elements[pos]
            pos += 1
            continue
        
        if vital_signs.oxygen_saturation is None and re.match(r'^[oO]2[:=.][0-9]+$', elements[pos]):
            vital_signs.oxygen_saturation = elements[pos][3:]
            pos += 1
            continue
        
        if vital_signs.oxygen_saturation is None and re.match(r'^[0-9]+%.*$', elements[pos]): # Ends with %
            vital_signs.oxygen_saturation = extract_numbers(elements[pos]) + "%"
            pos += 1
            continue

        # Corrected regular expression pattern
        if vital_signs.oxygen_saturation is None and re.match(r'^[0-9]+%?\(?[Rr][Aa]\)?.*$', elements[pos]):  # Ends with "RA"
            vital_signs.oxygen_saturation = extract_numbers(elements[pos]) + "%"
            pos += 1

        # Corrected regular expression pattern
        if vital_signs.oxygen_saturation is None and re.match(r'^[Ss]at:[0-9]+%?\(?[Rr][Aa]\)?.*$', elements[pos]):
            vital_signs.oxygen_saturation = extract_numbers(elements[pos][4:]) + "%"
            pos += 1


        if vital_signs.oxygen_saturation is None and re.match(r'^[Ss]a[Oo]2[0-9]+%?$', elements[pos]):
            vital_signs.oxygen_saturation = extract_numbers(elements[pos][4:]) + "%"
            pos += 1
            continue
        
        if vital_signs.oxygen_saturation is None and elements[pos] == "%":
            vital_signs.oxygen_saturation = elements[pos-1] + "%"
        
        if vital_signs.oxygen_saturation is None and vital_signs.respiration_rate is not None and elements[pos] == "RA":
            vital_signs.oxygen_saturation = elements[pos-1] + "%"

        if re.match(r'^[0-9]+$', elements[pos]):  # Just numbers
            value = int(elements[pos])
            if vital_signs.respiration_rate is None and 4 < value < 30:  # Respiration rate
                vital_signs.respiration_rate = elements[pos]
                pos += 1
                continue
            elif vital_signs.heart_rate is None and 30 <= value < 200:  # Heart rate
                vital_signs.heart_rate = elements[pos]
                pos += 1
                continue
        
        if re.match(r'^[0-9]+-[0-9]+$', elements[pos]):  # Two numbers separated by hyphen
            parts = elements[pos].split('-')
            value = int(parts[0])
            if vital_signs.respiration_rate is None and 4 < value < 30:  # Respiration rate
                vital_signs.respiration_rate = elements[pos]
                pos += 1
                continue
            elif vital_signs.heart_rate is None and 30 <= value < 200:  # Heart rate
                vital_signs.heart_rate = elements[pos]
                pos += 1
                continue
        
        if re.match(r'^[0-9]+/[0-9]+$', elements[pos]):
            vital_signs.blood_pressure = elements[pos]
            pos += 1
            continue
        
        if re.match(r'^[0-9]+-[0-9]+/[0-9]+-[0-9]+$', elements[pos]):
            vital_signs.blood_pressure = elements[pos]
            pos += 1
            continue
        
        pos += 1
        
    return vital_signs