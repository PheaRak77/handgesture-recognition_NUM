-- Seed all gestures from src/gesture_engine.py (GESTURE_KHMER)
-- After editing code, also run: python scripts/sync_gestures.py

INSERT INTO gestures (name_en, text_khmer, text_english, gesture_type) VALUES
('No Hand',       'គ្មានដៃ',        'No hand detected',           'single_hand'),
('Fist',          'ក្ដាប់ដៃ',       'Fist / Stop',                'single_hand'),
('Open Hand',     'ដៃបើក',         'Open hand / Hello',          'single_hand'),
('Thumbs Up',     'ល្អណាស់',        'Thumbs Up / Good',           'single_hand'),
('Thumbs Down',   'មិនល្អ',         'Thumbs Down / Bad',          'single_hand'),
('One',           'មួយ',           'Number One / Index finger',  'single_hand'),
('Two',           'ពីរ',            'Number Two',                 'single_hand'),
('Three',         'បី',            'Number Three',               'single_hand'),
('Four',          'បួន',           'Number Four',                'single_hand'),
('Five',          'ប្រាំ',          'Number Five',                'single_hand'),
('OK',            'យល់ព្រម',        'OK / Agree',                 'single_hand'),
('Call Me',       'ហៅខ្ញុំ',        'Call me / Phone',            'single_hand'),
('Point Up',      'ចង្អុលឡើង',      'Point Up / Attention',       'single_hand'),
('Hello',         'សួស្តី',         'Hello / Greeting',           'two_hand'),
('How Are You',   'សុខសប្បាយ',      'How are you?',               'single_hand'),
('Where From',    'មកពីណា',        'Where are you from?',        'two_hand'),
('Thank You',     'អរគុណ',         'Thank you',                  'two_hand'),
('Please',        'សូម',           'Please',                     'two_hand'),
('Sorry',         'សុំទោស',        'Sorry',                      'single_hand'),
('Right',         'ត្រូវ',          'Right',                      'two_hand'),
('Wrong',         'ខុស',            'Wrong',                      'single_hand'),
('Understand',    'យល់',            'Understand',                 'single_hand'),
('Again',         'ម្តងទៀត',        'Again',                      'two_hand'),
('Deaf',          'ថ្លង់',           'Deaf',                       'single_hand'),
('Congratulation','អបអរសាទរ',       'Congratulation',             'two_hand'),
('Hearing',       'ស្តាប់ឮ',         'Hearing',                    'single_hand')
ON CONFLICT (name_en) DO UPDATE SET
    text_khmer   = EXCLUDED.text_khmer,
    text_english = EXCLUDED.text_english,
    gesture_type = EXCLUDED.gesture_type,
    is_active    = TRUE,
    updated_at   = NOW();
