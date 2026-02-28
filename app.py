
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
import re
import time
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Get Groq API key from environment
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY not found in .env file")
    print("Please add GROQ_API_KEY=your_api_key_here to your .env file")

# Initialize Groq client
client = None
if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)

# Global dataframe storage
df = None

def detect_columns_used(code, available_columns):
    """Detect which dataframe columns are referenced in the generated code."""
    columns_used = []
    for col in available_columns:
        # Check for column references: df['col'], df["col"], or df.col
        if f"['{col}']" in code or f'["{col}"]' in code or f".{col}" in code:
            columns_used.append(col)
    return columns_used

def detect_analysis_type(code):
    """Detect the type of analysis based on code patterns."""
    code_lower = code.lower()
    
    if 'crosstab' in code_lower:
        return 'Cross-tabulation'
    elif 'groupby' in code_lower:
        return 'Grouping'
    elif 'df[' in code and any(op in code for op in ['.sum()', '.mean()', '.max()', '.min()', '.count()']):
        return 'Filtering + Aggregation'
    elif 'df[' in code:
        return 'Filtering'
    elif any(op in code for op in ['.sum()', '.mean()', '.max()', '.min()', '.std()', '.median()']):
        return 'Aggregation'
    elif any(keyword in code_lower for keyword in ['cut', 'qcut', 'bin']):
        return 'Segmentation'
    else:
        return 'Descriptive'

def calculate_complexity(code):
    """Calculate complexity level based on code structure."""
    complexity_score = 0
    
    # Count complexity indicators
    if 'groupby' in code:
        complexity_score += 2
    if 'df[' in code:
        complexity_score += 1
    if 'lambda' in code or '.apply(' in code:
        complexity_score += 2
    if '&' in code or '|' in code:  # Multiple conditions
        complexity_score += 1
    if any(op in code for op in ['.merge(', '.join(', '.pivot']):
        complexity_score += 3
    
    if complexity_score >= 4:
        return 'Advanced'
    elif complexity_score >= 2:
        return 'Intermediate'
    else:
        return 'Basic'

def calculate_confidence_score(code, had_error=False):
    """Calculate confidence score based on analysis complexity."""
    if had_error:
        return 0
    
    confidence = 100
    
    # Reduce confidence based on complexity
    if 'df[' in code:
        confidence -= 5  # Filtering applied
    if 'groupby' in code:
        confidence -= 10  # Grouping adds complexity
    if 'lambda' in code or '.apply(' in code:
        confidence -= 15  # Custom functions reduce certainty
    if '&' in code or '|' in code:
        confidence -= 5  # Multiple conditions
    
    return max(confidence, 0)

def calculate_rows_processed(code, result, total_rows):
    """Estimate rows processed based on result type."""
    # If result is a filtered DataFrame, count its rows
    if isinstance(result, pd.DataFrame):
        return len(result)
    # If filtering is detected but result is not a DataFrame
    elif 'df[' in code:
        # Try to estimate, default to total if uncertain
        return total_rows
    else:
        # No filtering, all rows processed
        return total_rows

def generate_data_passport(dataframe):
    """
    Generate a data passport with metadata about the uploaded CSV.
    
    Args:
        dataframe: pandas DataFrame
        
    Returns:
        dict: Data passport containing rows, columns, and column details
    """
    passport = {
        'total_rows': int(dataframe.shape[0]),
        'total_columns': int(dataframe.shape[1]),
        'columns': []
    }
    
    for column in dataframe.columns:
        col_info = {
            'name': str(column),
            'dtype': str(dataframe[column].dtype),
            'unique_values': int(dataframe[column].nunique()),
            'sample_values': dataframe[column].dropna().unique()[:3].tolist()
        }
        passport['columns'].append(col_info)
    
    return passport

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/preview', methods=['GET'])
def preview_data():
    """Preview uploaded CSV data (first 10 rows)."""
    global df
    
    try:
        if df is None:
            return jsonify({
                'status': 'error',
                'message': 'No CSV file uploaded yet'
            }), 400
        
        # Get first 10 rows
        preview_df = df.head(10)
        
        # Convert to dict with columns and data
        preview_data = {
            'columns': preview_df.columns.tolist(),
            'data': preview_df.values.tolist()
        }
        
        return jsonify({
            'status': 'success',
            'preview': preview_data
        }), 200
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error previewing data: {str(e)}'
        }), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle CSV file upload and generate data passport.
    
    Returns:
        JSON response with success message and data passport
    """
    global df
    
    try:
        # Check if file is present in request
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        # Check if filename is empty
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'No file selected'
            }), 400
        
        # Check if file is CSV
        if not file.filename.endswith('.csv'):
            return jsonify({
                'status': 'error',
                'message': 'Invalid file format. Please upload a CSV file'
            }), 400
        
        # Read CSV file using pandas
        df = pd.read_csv(file)
        
        # Check if dataframe is empty
        if df.empty:
            return jsonify({
                'status': 'error',
                'message': 'CSV file is empty'
            }), 400
        
        # Generate data passport
        passport = generate_data_passport(df)
        
        return jsonify({
            'status': 'success',
            'message': 'Upload successful',
            'passport': passport
        }), 200
    
    except pd.errors.EmptyDataError:
        return jsonify({
            'status': 'error',
            'message': 'CSV file is empty or corrupted'
        }), 400
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error processing file: {str(e)}'
        }), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """
    Handle natural language questions about the data using Groq.
    
    Returns:
        JSON response with answer and generated code
    """
    global df
    
    try:
        # Check if API key is configured
        if not GROQ_API_KEY or not client:
            return jsonify({
                'status': 'error',
                'message': 'Groq API key not configured. Please add GROQ_API_KEY to .env file'
            }), 500
        
        # Check if dataframe exists
        if df is None:
            return jsonify({
                'status': 'error',
                'message': 'Please upload a CSV file first'
            }), 400
        
        # Get question from request
        data = request.get_json()
        question = data.get('question', '')
        
        if not question:
            return jsonify({
                'status': 'error',
                'message': 'No question provided'
            }), 400
        
        # Build prompt for Groq - expanded version (MAX_PROMPT_TOKENS = 4000)
        columns_info = f"Columns: {', '.join(df.columns.tolist())}"
        prompt = f"""
You are an expert Python Data Analyst operating inside a secure Conversational Data Intelligence Platform.

Your responsibility is to translate a user's natural language analytical question into VALID, EXECUTABLE pandas code using a provided dataframe named `df`.

You are NOT a chatbot.  
You are a deterministic analytics engine whose output will be executed automatically inside a controlled Python environment.

Your output must strictly follow the rules below.

==================================================
CORE OBJECTIVE
==================================================

Generate executable pandas expressions that compute answers ONLY from the provided dataset.

The generated code will be executed using eval(), therefore correctness, safety, and determinism are critical.

You must produce reliable analytical logic grounded strictly in the dataset schema.

==================================================
DATA CONTEXT
==================================================

The dataframe variable name is:

    df

Available dataset schema:

{columns_info}

You MUST use only the columns listed above.

Never assume additional columns exist.
Never invent column names.
Never rely on external data or knowledge.

==================================================
STRICT OUTPUT FORMAT RULES
==================================================

Your response MUST:

- Contain ONLY executable Python pandas code.
- NOT include explanations.
- NOT include markdown formatting.
- NOT include comments.
- NOT include natural language text.
- NOT include backticks.
- NOT include imports.
- NOT include function definitions.
- NOT include print().
- NOT include return statements.
- NOT assign variables.
- NOT modify the dataframe.
- NOT create new files or access system resources.

The FINAL LINE must be a pandas expression that directly evaluates to the result.

Correct examples:

    len(df)

    df['MonthlyCharges'].mean()

    df.groupby('gender')['Churn'].apply(lambda x: (x == 'Yes').mean()*100)

Incorrect examples:

    print(df.head())
    return df.mean()
    result = df.sum()
    ```python ... ```

==================================================
ANALYTICAL SCOPE RULES
==================================================

You may ONLY perform DESCRIPTIVE analytics on EXISTING data.

Allowed:
- aggregation (sum, mean, count, etc.)
- filtering existing records
- grouping and summarization
- comparison of existing values
- percentage calculations from existing data
- ranking existing records
- segmentation of existing data
- summary statistics

NOT allowed:
- predictions (likely, will, future, forecast, predict, estimate future outcomes)
- forecasting
- causal explanations (why something happened)
- recommendations (should, suggest, advise)
- assumptions about future behavior
- external comparisons
- industry benchmarks
- machine learning inferences

CRITICAL DISTINCTION - Predictive vs Descriptive:

PREDICTIVE (NOT ALLOWED):
- "which customers are likely to leave" → requires ML model
- "predict sales for next month" → requires forecasting
- "who will churn" → requires prediction model
- "estimate future revenue" → requires forecasting
- "which customers might cancel" → requires prediction

DESCRIPTIVE (ALLOWED):
- "show customers who have already left" → df[df['Churn'] == 'Yes']
- "count customers who churned" → len(df[df['Churn'] == 'Yes'])
- "what is the historical churn rate" → (df['Churn'] == 'Yes').mean()
- "show characteristics of churned customers" → descriptive filtering

If a question uses predictive language (likely, will, predict, forecast, estimate future, might, could, should):

IMMEDIATELY produce:

    ValueError("This question appears to require predictive analysis, which is not supported. This platform performs descriptive analytics on existing data only. Please rephrase to analyze historical or current data (e.g., 'show customers who have already churned' instead of 'which customers are likely to churn').")

==================================================
COLUMN VALIDATION RULE
==================================================

If the user references a column not present in the schema:

DO NOT guess.

Instead produce:

    ValueError("Requested column not found in dataset")

==================================================
TIME SERIES SAFETY RULE
==================================================

Only perform time-based analysis IF a datetime column exists.

You MUST NOT:
- use df.index for dates
- assume month/year columns
- extract .month or .year unless datatype is datetime

If no datetime column exists but user requests trend/month/year analysis:

Produce:

    ValueError("No datetime column available for temporal analysis")

==================================================
ANALYTICAL INTERPRETATION GUIDELINES
==================================================

Interpret business language carefully, distinguishing descriptive from predictive:

DESCRIPTIVE (Valid):
- "count" → len(df) or filtering + count
- "average" / "mean" → .mean()
- "total" / "sum" → .sum()
- "maximum" → .max()
- "minimum" → .min()
- "unique" → .nunique()
- "show customers who churned" → df[df['Churn'] == 'Yes']
- "characteristics of high spenders" → df[df['MonthlyCharges'] > threshold]
- "historical churn rate" → (df['Churn'] == 'Yes').mean()

PREDICTIVE (Invalid - return ValueError):
- "likely to churn" → predictive, not allowed
- "will leave" → predictive, not allowed
- "might cancel" → predictive, not allowed
- "predict revenue" → predictive, not allowed
- "forecast sales" → predictive, not allowed

Rates and percentages from existing data:

Overall rate:
    (condition_count / len(df)) * 100

Grouped rate:
    df.groupby('column').apply(lambda x: (condition / len(x))*100)

Filtering existing records:
    df[df['column'] == value]

Multiple conditions:
    (cond1) & (cond2)

Sorting:
    .sort_values(ascending=False)

Top N:
    .head(N)

==================================================
AMBIGUITY HANDLING
==================================================

If the question is vague or undefined
(example: "show performance", "give insights", "analyze the data"):

Produce:

    ValueError("Question lacks specificity. Please provide a clear analytical question about specific columns or metrics in the dataset.")

If the question contains predictive keywords without clear descriptive intent:

Keywords: likely, will, predict, forecast, estimate (future), might, could, should, recommend

Produce:

    ValueError("This question appears to require predictive analysis, which is not supported. This platform performs descriptive analytics on existing data only. Please rephrase to analyze historical or current data.")

==================================================
SECURITY CONSTRAINTS
==================================================

Never generate code containing:

- import
- open(
- exec(
- eval(
- os.
- sys.
- subprocess
- file operations
- network calls
- deletion commands

You operate ONLY on dataframe df.

==================================================
DATA EXPOSURE RULE
==================================================

Do NOT output entire dataset.

If user requests full data display,
return a safe summary:

    df.describe(include='all')

==================================================
LOGICAL CONSISTENCY RULE
==================================================

Ensure generated code:

- references valid columns
- uses correct pandas syntax
- produces deterministic output
- does not rely on randomness
- executes without assignment statements

==================================================
USER QUESTION
==================================================

{question}

==================================================
FINAL INSTRUCTION
==================================================

Return ONLY executable pandas code.

No explanations.
No formatting.
No text outside Python code.
"""
        
        # Call Groq model
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract response text
        generated_code = response.choices[0].message.content.strip()
        
        # Clean markdown if present
        generated_code = re.sub(r'^```python\s*\n', '', generated_code, flags=re.MULTILINE)
        generated_code = re.sub(r'^```\s*\n', '', generated_code, flags=re.MULTILINE)
        generated_code = re.sub(r'\n```\s*$', '', generated_code, flags=re.MULTILINE)
        generated_code = generated_code.strip()
        
        # Clean up return statements (not needed in exec/eval context)
        # Replace "return expression" with just "expression"
        generated_code = re.sub(r'^\s*return\s+', '', generated_code, flags=re.MULTILINE)
        
        # Execute the generated code
        # Create a safe set of built-in functions
        safe_builtins = {
            'len': len,
            'sum': sum,
            'min': min,
            'max': max,
            'abs': abs,
            'round': round,
            'int': int,
            'float': float,
            'str': str,
            'bool': bool,
            'list': list,
            'dict': dict,
            'set': set,
            'tuple': tuple,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'sorted': sorted,
            'any': any,
            'all': all,
            # Exception types
            'ValueError': ValueError,
            'TypeError': TypeError,
            'KeyError': KeyError,
            'IndexError': IndexError,
            'AttributeError': AttributeError,
            'Exception': Exception,
        }
        
        namespace = {'df': df, 'pd': pd, '__builtins__': safe_builtins}
        
        # Initialize transparency tracking
        execution_error = False
        start_time = time.time()
        
        try:
            # Check if it's a single expression (can use eval)
            # or multiple statements (need exec)
            try:
                # Try eval first for simple expressions
                result = eval(generated_code, namespace)
            except SyntaxError:
                # If eval fails, use exec for statements
                # Wrap the code to capture the last line's result
                lines = generated_code.strip().split('\n')
                if len(lines) > 1:
                    # Multi-line code: execute all but last, then eval last line
                    exec('\n'.join(lines[:-1]), namespace)
                    result = eval(lines[-1], namespace)
                else:
                    # Single line statement - execute and check namespace for result
                    exec(generated_code, namespace)
                    # Try to find a result variable or use the modified df
                    result = namespace.get('result', df)
            
            # Detect if result can be visualized as a chart
            chart_data = None
            
            # Check if result is a pandas Series with numeric values
            if isinstance(result, pd.Series):
                # Check if values are numeric
                if pd.api.types.is_numeric_dtype(result):
                    labels = result.index.tolist()
                    values = result.values.tolist()
                    
                    # Determine available chart types based on data characteristics
                    available_charts = ['bar', 'line']
                    
                    # Add pie/doughnut if values are positive and <= 10 categories
                    if all(v >= 0 for v in values) and len(values) <= 10:
                        available_charts.extend(['pie', 'doughnut'])
                    
                    # Add horizontal bar for many categories
                    if len(values) > 5:
                        available_charts.append('horizontalBar')
                    
                    chart_data = {
                        'labels': labels,
                        'values': values,
                        'available_types': available_charts,
                        'default_type': 'bar'
                    }
            
            # Check if result is a DataFrame with 2 columns
            elif isinstance(result, pd.DataFrame) and len(result.columns) == 2:
                # Find categorical and numeric columns
                col1, col2 = result.columns[0], result.columns[1]
                
                labels = None
                values = None
                
                # Check if one column is numeric
                if pd.api.types.is_numeric_dtype(result[col2]):
                    labels = result[col1].tolist()
                    values = result[col2].tolist()
                elif pd.api.types.is_numeric_dtype(result[col1]):
                    labels = result[col2].tolist()
                    values = result[col1].tolist()
                
                if labels and values:
                    # Determine available chart types
                    available_charts = ['bar', 'line']
                    
                    # Add pie/doughnut if values are positive and <= 10 categories
                    if all(v >= 0 for v in values) and len(values) <= 10:
                        available_charts.extend(['pie', 'doughnut'])
                    
                    # Add horizontal bar for many categories
                    if len(values) > 5:
                        available_charts.append('horizontalBar')
                    
                    # Add area chart option
                    available_charts.append('area')
                    
                    chart_data = {
                        'labels': labels,
                        'values': values,
                        'available_types': available_charts,
                        'default_type': 'bar'
                    }
            
            # Convert result to string for JSON serialization
            # MAX_RESULT_FOR_EXPLANATION = 2000 tokens (~8000 characters)
            if isinstance(result, pd.DataFrame):
                result_str = result.to_string()
                # For explanation, limit to first 100 rows (more context for better explanations)
                result_str_truncated = result.head(100).to_string()
                if len(result) > 100:
                    result_str_truncated += f"\n... ({len(result) - 100} more rows)"
                # Ensure we don't exceed 8000 characters
                if len(result_str_truncated) > 8000:
                    result_str_truncated = result_str_truncated[:8000] + "\n... (truncated)"
            elif isinstance(result, pd.Series):
                result_str = result.to_string()
                # For explanation, limit to first 100 items
                result_str_truncated = result.head(100).to_string()
                if len(result) > 100:
                    result_str_truncated += f"\n... ({len(result) - 100} more items)"
                # Ensure we don't exceed 8000 characters
                if len(result_str_truncated) > 8000:
                    result_str_truncated = result_str_truncated[:8000] + "\n... (truncated)"
            else:
                result_str = str(result)
                result_str_truncated = result_str[:8000]  # Limit to 8000 chars (2000 tokens)
        
        except ValueError as value_error:
            # ValueError is intentionally raised for predictive/invalid questions
            # Return a clean, professional error message to the user
            execution_error = True
            error_message = str(value_error)
            
            # Generate transparency for error case
            transparency = {
                'data_source': 'Uploaded CSV',
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'columns_used': detect_columns_used(generated_code, df.columns.tolist()),
                'rows_processed': 0,
                'analysis_type': 'Invalid',
                'complexity': 'N/A',
                'external_data_used': False,
                'confidence_score': 0,
                'execution_time_ms': round((time.time() - start_time) * 1000, 2)
            }
            
            return jsonify({
                'status': 'info',
                'message': error_message,
                'code': generated_code,
                'transparency': transparency
            }), 200
        
        except Exception as exec_error:
            execution_error = True
            
            # Generate transparency for execution error
            transparency = {
                'data_source': 'Uploaded CSV',
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'columns_used': detect_columns_used(generated_code, df.columns.tolist()),
                'rows_processed': 0,
                'analysis_type': 'Error',
                'complexity': 'N/A',
                'external_data_used': False,
                'confidence_score': 0,
                'execution_time_ms': round((time.time() - start_time) * 1000, 2)
            }
            
            return jsonify({
                'status': 'error',
                'message': f'Error executing generated code: {str(exec_error)}',
                'code': generated_code,
                'transparency': transparency
            }), 500
        
        # Record execution time
        execution_time_ms = round((time.time() - start_time) * 1000, 2)
        
        # Generate transparency metadata
        columns_used = detect_columns_used(generated_code, df.columns.tolist())
        analysis_type = detect_analysis_type(generated_code)
        complexity = calculate_complexity(generated_code)
        confidence_score = calculate_confidence_score(generated_code, execution_error)
        rows_processed = calculate_rows_processed(generated_code, result, len(df))
        
        transparency = {
            'data_source': 'Uploaded CSV',
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns_used': columns_used,
            'rows_processed': rows_processed,
            'analysis_type': analysis_type,
            'complexity': complexity,
            'external_data_used': False,
            'confidence_score': confidence_score,
            'execution_time_ms': execution_time_ms
        }
        
        # Generate natural language answer and explanation
        explanation_prompt = f"""You are providing insights about data analysis results.

User Question: {question}

Generated Code: {generated_code}

Result: {result_str_truncated}

Provide TWO responses:

1. ANSWER: A direct, business-friendly answer to the user's question in 3-4 sentences. MUST include specific numerical values, counts, percentages, or key data points from the Result above. Be concrete and reference actual computed values. Focus on WHAT the data shows with exact numbers.

2. EXPLANATION: A technical explanation of what the code does and how it computes the result in 2-3 sentences.

Format your response EXACTLY like this:
ANSWER: [Your business-focused answer with specific numbers from the Result]
EXPLANATION: [Your technical code explanation here]"""
        
        # Call Groq for natural language answer and explanation
        try:
            explanation_response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "user", "content": explanation_prompt}
                ]
            )
            
            # Extract the response and parse into answer and explanation
            full_response = explanation_response.choices[0].message.content.strip()
            
            # Parse the response to extract ANSWER and EXPLANATION
            answer = ""
            explanation = ""
            
            # Try to extract ANSWER and EXPLANATION using regex
            answer_match = re.search(r'ANSWER:\s*(.+?)(?=EXPLANATION:|$)', full_response, re.DOTALL | re.IGNORECASE)
            explanation_match = re.search(r'EXPLANATION:\s*(.+?)$', full_response, re.DOTALL | re.IGNORECASE)
            
            if answer_match:
                answer = answer_match.group(1).strip()
            if explanation_match:
                explanation = explanation_match.group(1).strip()
            
            # Fallback if parsing fails
            if not answer:
                answer = full_response
                explanation = "Analysis completed successfully."
        
        except Exception as explanation_error:
                # If explanation fails (e.g., rate limit), use the raw result
                error_msg = str(explanation_error)
                if 'rate_limit' in error_msg.lower() or '413' in error_msg:
                    # Rate limit hit - return result without explanation
                    answer = f"Result: {result_str_truncated}"
                    explanation = "Unable to generate explanation due to rate limit."
                else:
                    # Other error - still return result with error note
                    answer = f"Result: {result_str_truncated}"
                    explanation = f"Unable to generate explanation due to API error."
        
        except Exception as exec_error:
            return jsonify({
                'status': 'error',
                'message': f'Error executing generated code: {str(exec_error)}',
                'code': generated_code
            }), 500
        
        return jsonify({
            'status': 'success',
            'answer': answer,
            'explanation': explanation,
            'code': generated_code,
            'chart': chart_data,
            'transparency': transparency
        }), 200
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error processing question: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
