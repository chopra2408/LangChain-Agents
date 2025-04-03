import os
import time
import json
import logging
from typing import Dict, List
from tavily import TavilyClient
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tiktoken import get_encoding
from langchain_core.messages import BaseMessage # Import BaseMessage
from pydantic import BaseModel, Field, ValidationError
from langchain.output_parsers import PydanticOutputParser
from typing import List, Dict
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import json
# Remove this if NutritionAgent handles its own streaming printout
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- Pydantic Schema and Parser Definition (Module Level) ---
class NutritionResponseSchema(BaseModel):
    title: str = Field(..., description="User's query accurately reflected as the title")
    summary: str = Field(..., description="A concise (4-5 sentences) summary of the key findings and recommendations based *only* on the provided research data")
    sources: List[Dict[str, str]] = Field(..., description="List of sources used from the RESEARCH DATA, each with 'title' and 'url' keys. Only include sources explicitly listed in the input RESEARCH DATA.")
    content: List[str] = Field(..., description="Detailed main content, broken down into logical markdown-formatted strings. Each string should represent a paragraph or section (e.g., '## Key Findings', '### Nutrient Breakdown', '## Practical Recommendations', '## Potential Risks'). Use headings (##, ###), lists (*, -), and bold text (**important**). Base content *strictly* on the provided RESEARCH DATA.")
    related_questions: List[str] = Field(..., min_items=5, max_items=5, description="Exactly 5 relevant follow-up questions a user might ask based on the generated content and the initial query.")

# Instantiate the parser globally or where it's needed before PromptTemplate
parser = PydanticOutputParser(pydantic_object=NutritionResponseSchema)
# --- End Schema/Parser Definition ---


class MessageHistory:
    # --- MessageHistory class remains the same ---
    """Class for managing conversation history"""
    def __init__(self, max_history: int = 10):
        self.history = []
        self.max_history = max_history

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_history(self):
        return self.history

    def get_formatted_history(self):
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.history])


class NutritionAgent:
    """Agent for handling nutrition and health queries with web search"""
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.tavily_client = TavilyClient(api_key=api_key)
        self.chat_model = ChatOpenAI(
            model=model,
            # REMOVE callbacks if manually printing stream in generate_nutrition_response
            # callbacks=[StreamingStdOutCallbackHandler()],
            temperature=0.4,
            max_tokens=2500, # Increased slightly for JSON verbosity
            # Remove model_kwargs if relying on PydanticOutputParser format instructions
            # model_kwargs={"response_format": {"type": "json_object"}},
            streaming=True
        )
        self.encoding = get_encoding("cl100k_base")

        # --- Define Template String and PromptTemplate here ---
        template_string = """
        **Role:** You are 'FitMentor', an expert AI assistant specializing in nutrition and health sciences. Your credentials include advanced knowledge in dietetics and exercise physiology. Your responses must be scientifically accurate, objective, detailed, and actionable, based *strictly* on the provided research data.

        **Task:** Generate a structured JSON response summarizing and detailing findings from the provided research data in relation to the user's query. Adhere strictly to the output format specified.

        **Input Data:**

        1.  **RESEARCH DATA:**
            ```json
            {research_summary}
            ```

        2.  **USER QUERY:**
            "{query}"

        **Instructions:**

        1.  **Analyze:** Carefully examine the `RESEARCH DATA` in the context of the `USER QUERY`. Identify key findings, data points, recommendations, potential risks, and source information present in the data.
        2.  **Structure Content:** Organize the extracted information logically within the `content` field of the output schema. Use markdown formatting effectively:
            *   Use headings (e.g., `## Key Findings`, `### Specific Benefits`, `## Practical Recommendations`, `## Potential Risks & Considerations`) to structure the information clearly.
            *   Use bullet points (`*` or `-`) for lists (e.g., listing recommendations, benefits, or risks).
            *   Use bold text (`**text**`) for critical warnings or important takeaways.
            *   Each element in the `content` list should be a self-contained markdown string representing a paragraph or section.
        3.  **Populate Fields:** Fill all fields of the specified output schema accurately:
            *   `title`: Reflect the original `USER QUERY`.
            *   `summary`: Provide a brief (4-5 sentence) overview of the main points derived *only* from the `RESEARCH DATA`.
            *   `sources`: List *only* the sources provided in the input `RESEARCH DATA`, extracting their titles and URLs.
            *   `content`: Include the detailed, markdown-formatted information as described in Instruction 2. Ensure this content is derived *solely* from the `RESEARCH DATA`.
            *   `related_questions`: Generate exactly 5 relevant follow-up questions based on the generated content.
        4.  **Strict Formatting:** Your *entire* output must be a single, valid JSON object conforming precisely to the schema described below. Do not include *any* introductory text, concluding remarks, or markdown formatting outside the JSON structure itself (e.g., no ```json ... ``` markers around the final output).

        **Required Output Format:**

        {format_instructions}

        **Constraint Checklist:**
        *   Is the entire output a single JSON object? YES/NO
        *   Does the JSON object perfectly match the provided schema structure? YES/NO
        *   Is all information derived *only* from the provided `RESEARCH DATA`? YES/NO
        *   Are the sources listed *only* those from the `RESEARCH DATA` input? YES/NO
        *   Does the `content` field use markdown formatting correctly? YES/NO
        *   Are there exactly 5 `related_questions`? YES/NO
        *   Is there *any* text outside the main JSON object? YES/NO (Should be NO)

        Proceed with generating the JSON output based *only* on the provided data and instructions.
        """

        self.nutrition_prompt = PromptTemplate(
            template=template_string,
            input_variables=["query", "research_summary"],
            partial_variables={"format_instructions": parser.get_format_instructions()} # Use the globally defined parser
        )
        # --- End PromptTemplate Definition ---


    def count_tokens(self, text: str) -> int:
        """Return the token count for a given text using the provided encoding."""
        return len(self.encoding.encode(text))

    # safe_api_invoke might not be needed if using .stream directly, but keep if useful elsewhere
    def safe_api_invoke(self, messages: List[BaseMessage], max_retries=3, retry_delay=2):
         """Invoke the API safely with retries on rate limit errors."""
         # This is designed for .invoke(), not easily adaptable to .stream() retries
         # Consider adding retry logic around the stream iteration if needed
         for attempt in range(max_retries):
            try:
                response = self.chat_model.invoke(messages) # Use invoke for retry logic
                return response
            except Exception as e:
                logging.warning(f"API call failed on attempt {attempt+1}/{max_retries}: {e}")
                time.sleep(retry_delay * (attempt + 1))
                if attempt == max_retries - 1:
                    raise Exception(f"Exceeded maximum retry attempts for API invocation: {e}")


    def search_for_information(self, query: str) -> Dict:
        # --- search_for_information remains the same ---
        logging.info(f"Researching nutrition query: {query}")
        try:
            search_results = self.tavily_client.search(query=query, max_results=7, search_depth="advanced", include_answer=True, include_raw_content=False, time_range="month", include_domains=["nih.gov", "mayoclinic.org", "health.harvard.edu", "medicalnewstoday.com", "healthline.com", "webmd.com", "ncbi.nlm.nih.gov", "hopkinsmedicine.org"])['results']
            logging.info(f"Found {len(search_results)} general sources.")
            research_summary = {"general_results": [{"url": r["url"], "title": r["title"], "content": r.get("snippet", "")} for r in search_results],}
            return research_summary
        except Exception as e:
            logging.error(f"Error during web search: {e}")
            return {"general_results": [], "error": str(e)}


    def generate_nutrition_response(self, query: str, research_summary: Dict) -> NutritionResponseSchema:
        """
        Generate a structured nutrition response using PromptTemplate, stream it manually,
        and validate it against the Pydantic schema.
        """

        # --- Format the prompt using the instance's PromptTemplate ---
        try:
             # Ensure research_summary is a JSON string for the template
            research_summary_str = json.dumps(research_summary, indent=2)
            formatted_prompt = self.nutrition_prompt.format_prompt(
                query=query,
                research_summary=research_summary_str
            )
            messages = formatted_prompt.to_messages()
            prompt_string_for_tokens = formatted_prompt.to_string() # For token counting
        except Exception as e:
            logging.error(f"Error formatting prompt: {e}")
            raise ValueError(f"Failed to format prompt: {e}") from e
        # --- End Prompt Formatting ---


        # --- Token management ---
        total_input_tokens = self.count_tokens(prompt_string_for_tokens)
        max_output_tokens = 2500 # Define max expected output tokens
        # Correct model name check if needed, or use known context window
        model_context = getattr(self.chat_model, 'model_name', 'gpt-4o-mini')
        max_context_tokens = 128000 if 'gpt-4o-mini' in model_context else 8000

        if total_input_tokens + max_output_tokens > max_context_tokens:
             logging.warning(f"Input tokens ({total_input_tokens}) + max output ({max_output_tokens}) may exceed context limit ({max_context_tokens}). Truncating research data.")
             # Truncate research data
             original_research_summary = research_summary # Keep original if needed
             research_summary = self._truncate_research(
                 research_summary,
                 max_context_tokens - self.count_tokens(self.nutrition_prompt.template) - max_output_tokens - 200 # Estimate non-summary tokens + buffer
                )
             # Re-format the prompt with truncated data
             try:
                 research_summary_str = json.dumps(research_summary, indent=2)
                 formatted_prompt = self.nutrition_prompt.format_prompt(
                     query=query,
                     research_summary=research_summary_str
                 )
                 messages = formatted_prompt.to_messages()
                 prompt_string_for_tokens = formatted_prompt.to_string()
                 total_input_tokens = self.count_tokens(prompt_string_for_tokens) # Recalculate tokens
                 logging.info(f"Tokens after truncation: {total_input_tokens}")
                 if total_input_tokens + max_output_tokens > max_context_tokens:
                     logging.error("Input still too large even after truncation.")
                     raise ValueError("Input prompt too large for model context window after truncation.")
             except Exception as e:
                 logging.error(f"Error formatting prompt after truncation: {e}")
                 raise ValueError(f"Failed to format prompt after truncation: {e}") from e
        # --- End Token Management ---


        logging.info("Generating nutrition response and streaming...")
        start_time = time.time()

        accumulated_json_string = ""
        print("🌿 NutriAgent Stream: ", end="", flush=True) # Indicate start of stream display

        try:
            # Use .stream() with the formatted messages
            stream = self.chat_model.stream(messages)
            for chunk in stream:
                chunk_content = chunk.content
                if isinstance(chunk_content, str):
                    print(chunk_content, end="", flush=True) # Print chunk for streaming effect
                    accumulated_json_string += chunk_content # Accumulate the string

            print() # Newline after streaming finishes

            elapsed = time.time() - start_time
            logging.info(f"Stream finished in {elapsed:.2f} seconds. Validating response...")

            # --- Validation Step ---
            try:
                cleaned_json_string = accumulated_json_string.strip()
                # Optional: More robust cleaning if needed
                if cleaned_json_string.startswith("```json"):
                    cleaned_json_string = cleaned_json_string[7:]
                if cleaned_json_string.endswith("```"):
                    cleaned_json_string = cleaned_json_string[:-3]
                cleaned_json_string = cleaned_json_string.strip()

                if not cleaned_json_string:
                     raise ValueError("Received empty response after streaming.")

                # Use the global parser instance to validate
                validated_data = parser.parse(cleaned_json_string) # parser.parse for PydanticOutputParser

                logging.info("Response successfully parsed and validated against NutritionResponseSchema.")
                return validated_data # Return the validated Pydantic object

            except (json.JSONDecodeError, ValidationError, ValueError) as e: # Catch parsing/validation errors
                logging.error(f"Failed to parse or validate JSON response: {e}")
                logging.error(f"--- Raw LLM Output Start ---\n{accumulated_json_string}\n--- Raw LLM Output End ---")
                # Use the specific exception type from PydanticOutputParser if needed
                # from langchain_core.exceptions import OutputParserException
                # except OutputParserException as e: ...
                raise ValueError(f"LLM response failed validation: {e}. See logs for raw output.") from e

        except Exception as e:
            # Catch potential API errors or other issues during streaming
            logging.error(f"Error during API call or streaming: {e}")
            print() # Ensure newline if error happens mid-stream
            raise Exception(f"API call or streaming failed: {e}") from e


    def _truncate_research(self, research_summary: Dict, max_tokens: int) -> Dict:
        # --- _truncate_research remains the same ---
        """Truncate research data to fit within token limits"""
        truncated = {"general_results": []} # Simplified structure if needed
        current_token_count = self.count_tokens(json.dumps(truncated))
        base_prompt_tokens = max_tokens # Use the already calculated max_tokens allowed for summary

        # Add general results with remaining token budget
        items_added = 0
        for item in research_summary.get("general_results", []):
            # Estimate item tokens without full prompt formatting complexity
            item_str = json.dumps(item)
            item_tokens = self.count_tokens(item_str)

            if current_token_count + item_tokens < base_prompt_tokens:
                truncated["general_results"].append(item)
                current_token_count += item_tokens
                items_added += 1
            else:
                # Try adding just a snippet if content is too long
                if 'content' in item and item_tokens > 100: # Heuristic: only truncate long items
                    try:
                        # Estimate remaining tokens precisely
                        remaining_tokens = base_prompt_tokens - current_token_count
                        # Calculate how many chars fit roughly (adjust ratio as needed)
                        chars_to_keep = int(remaining_tokens * 3.5) # Approx chars per token
                        if chars_to_keep > 50 : # Minimum snippet length
                            snippet = item['content'][:chars_to_keep] + "..."
                            truncated_item = {k: v for k, v in item.items() if k != 'content'}
                            truncated_item['content'] = snippet
                            # Recheck tokens with the truncated item
                            item_str = json.dumps(truncated_item)
                            item_tokens = self.count_tokens(item_str)
                            if current_token_count + item_tokens < base_prompt_tokens:
                                truncated["general_results"].append(truncated_item)
                                current_token_count += item_tokens
                                items_added +=1
                                logging.info(f"Added truncated item: {item.get('title', 'N/A')}")
                            else:
                                logging.warning("Could not fit even truncated general item.")
                                break # Stop adding general results
                        else:
                             logging.warning("Not enough tokens remaining for a meaningful snippet.")
                             break
                    except Exception as e_trunc:
                         logging.warning(f"Error during snippet truncation: {e_trunc}")
                         break
                else:
                     logging.warning(f"Could not fit general item (short or no content): {item.get('title', 'N/A')}")
                     break # Stop adding general results

        if items_added == 0:
             logging.error("Truncation resulted in empty research data. Input might be too large or items too big.")
             # Return minimal structure to avoid downstream errors
             return {"error": "Input data too large after truncation", "general_results":[]}

        logging.info(f"Truncated research summary to {items_added} items.")
        return truncated


class ParentAI:
    # --- ParentAI class remains largely the same ---
    # Ensure its chat_model uses StreamingStdOutCallbackHandler if you want
    # general chat to stream automatically.
    """Parent AI that handles conversation routing and general queries"""
    def __init__(self):
        load_dotenv()
        self.tavily_api_key = os.environ.get('TAVILY_API_KEY')
        if not self.tavily_api_key:
            raise ValueError("TAVILY_API_KEY not found in environment variables.")

        self.nutrition_agent = NutritionAgent(api_key=self.tavily_api_key)
        self.conversation_history = MessageHistory()
        # Keep callback for general chat streaming
        from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler # Import here if needed
        self.chat_model = ChatOpenAI(
            model="gpt-4o-mini",
            callbacks=[StreamingStdOutCallbackHandler()],
            temperature=0.7,
            max_tokens=1000,
            streaming=True
        )

    def is_nutrition_query(self, query: str) -> bool:
        # --- is_nutrition_query remains the same ---
        system_prompt = ("You are a classifier that determines if a user query is related to nutrition, health, diet, supplements, fitness, or wellness. Respond with only 'YES' or 'NO'.")
        messages = [
            SystemMessage(content=system_prompt), # Correct: Initial instruction
            HumanMessage(content=f"Query: {query}\n...") # Correct: User input to classify
        ]
        try:
            # Use a non-streaming call for quick classification
            response = ChatOpenAI(model="gpt-4o-mini", temperature=0.0).invoke(messages)
            result = response.content.strip().upper()
            logging.info(f"Nutrition query classification for '{query}': {result}")
            return "YES" in result
        except Exception as e:
             logging.error(f"Error during nutrition query classification: {e}")
             return False # Default to False on error

    def handle_general_query(self, query: str) -> str:
        """Handle general conversation queries"""
        system_prompt = (
            "You are a helpful and friendly assistant. Provide conversational responses "
            "to general queries. Keep responses concise and natural."
        )

        # Get recent conversation history for context
        history = self.conversation_history.get_history()

        # Start with the main system prompt
        messages: List[BaseMessage] = [ # Use BaseMessage for type hint flexibility
            SystemMessage(content=system_prompt),
        ]

        # Add conversation history (last 5 messages)
        for msg in history[-5:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                # CORRECT: Use AIMessage for past assistant responses
                messages.append(AIMessage(content=msg["content"]))
                # INCORRECT was: messages.append(SystemMessage(content=f"Assistant: {msg['content']}"))

        # Add current query
        messages.append(HumanMessage(content=query))

        # The streaming happens via the callback attached in __init__
        # invoke triggers streaming callback and returns the final message
        response = self.chat_model.invoke(messages)

        return response.content
    def process_query(self, query: str) -> str:
        # --- process_query logic remains mostly the same ---
        # It correctly handles the validated object or error from generate_nutrition_response
        if query.lower() == '/bye':
            return "Goodbye! Shutting down..."

        if query.lower().startswith('/search '):
            search_query = query[8:].strip()
            if not search_query: return "Please provide a search query after /search."
            print("\n🌿 Conducting direct search...")
            try:
                research_data = self.nutrition_agent.search_for_information(search_query)
                if research_data.get("error"):
                    logging.error(f"Search failed: {research_data['error']}")
                    return f"Sorry, I couldn't complete the search due to an error: {research_data['error']}"

                # generate_nutrition_response handles its own streaming printout
                print("🌿 Generating structured search response (streaming...):")
                validated_response_object = self.nutrition_agent.generate_nutrition_response(search_query, research_data)

                timestamp = int(time.time())
                filename = f"search_response_{timestamp}.json"
                try:
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(validated_response_object.model_dump_json(indent=2))
                    logging.info(f"Validated search response saved to {filename}")
                    # Return summary
                    response_content = (f"**{validated_response_object.title}**\n\n"
                                        f"**Summary:** {validated_response_object.summary}\n\n"
                                        f"*(Full details were streamed and saved to {filename})*\n\n"
                                        f"**Related Questions:**\n" +
                                        "\n".join(f"- {q}" for q in validated_response_object.related_questions))
                    self.conversation_history.add_message("assistant", response_content) # Add response to history
                    return response_content
                except Exception as e_save:
                    logging.error(f"Error saving validated response to file {filename}: {e_save}")
                    response_content = f"Search result generated (error saving file):\n{validated_response_object.model_dump_json(indent=2)}"
                    self.conversation_history.add_message("assistant", response_content) # Add response to history
                    return response_content

            except (ValueError, Exception) as e:
                logging.error(f"Error in search processing: {e}")
                error_msg = f"I encountered an error during the search: {str(e)}. Please try again."
                if "LLM response failed validation" in str(e): error_msg = f"I found information, but couldn't structure it correctly. Error: {e}"
                elif "Search failed" in str(e): error_msg = f"Sorry, I couldn't complete the search. Error: {e}"
                self.conversation_history.add_message("assistant", error_msg) # Add error to history
                return error_msg

        # --- Regular Query Processing ---
        self.conversation_history.add_message("user", query)
        response_content: str
        is_nutrition = self.is_nutrition_query(query)

        if is_nutrition:
            logging.info("Detected nutrition query, routing to NutritionAgent")
            try:
                print("\n🌿 Researching nutrition information...")
                research_data = self.nutrition_agent.search_for_information(query)
                if research_data.get("error"):
                    logging.error(f"Search failed: {research_data['error']}")
                    response_content = f"Sorry, I couldn't complete the research due to an error: {research_data['error']}"
                else:
                    print("🌿 Generating nutrition response (streaming...):")
                    validated_response_object = self.nutrition_agent.generate_nutrition_response(query, research_data)
                    timestamp = int(time.time())
                    filename = f"nutrition_response_{timestamp}.json"
                    try:
                        with open(filename, "w", encoding="utf-8") as f:
                            f.write(validated_response_object.model_dump_json(indent=2))
                        logging.info(f"Validated nutrition response saved to {filename}")
                        response_content = (f"**{validated_response_object.title}**\n\n"
                                            f"**Summary:** {validated_response_object.summary}\n\n"
                                            f"*(Full details were streamed and saved to {filename}. Showing summary and related questions here.)*\n\n"
                                            f"**Related Questions:**\n" +
                                            "\n".join(f"- {q}" for q in validated_response_object.related_questions))
                    except Exception as e_save:
                        logging.error(f"Error saving validated response to file {filename}: {e_save}")
                        response_content = f"Nutrition response generated (error saving file):\n{validated_response_object.model_dump_json(indent=2)}"

            except (ValueError, Exception) as e:
                logging.error(f"Error in nutrition processing: {e}")
                if "LLM response failed validation" in str(e): response_content = f"I gathered nutrition information, but couldn't structure the response correctly. Error: {e}"
                elif "Search failed" in str(e): response_content = f"Sorry, I couldn't complete the research. Error: {e}"
                else: response_content = f"I encountered an error processing your nutrition query: {str(e)}. Please try asking differently."
        else:
            logging.info("Detected general query, handling with ParentAI")
            # General query streaming is handled by the callback in ParentAI's chat_model
            response_content = self.handle_general_query(query)

        self.conversation_history.add_message("assistant", response_content)
        return response_content


def main():
    # --- main function remains the same ---
    print("🌿 Welcome to NutriAgent - Your AI Nutrition and Health Assistant")
    print("Type '/bye' to exit. Use '/search [query]' for direct web search.")
    try:
        parent_ai = ParentAI()
        running = True
        while running:
            user_input = input("\n👤 You: ").strip()
            if not user_input: continue
            # Streaming printout happens *during* process_query via manual print or callback
            response = parent_ai.process_query(user_input)
            # Print the final summary/response string returned
            print(f"\n🌿 NutriAgent Final Response:\n{response}") # Clarify this is the final output
            if user_input.lower() == '/bye':
                running = False
    except KeyboardInterrupt:
        print("\n\n🌿 Session ended by user. Have a healthy day!")
    except Exception as e:
        logging.exception("Critical error in main loop:")
        print(f"\n💥 A critical error occurred: {e}")
        print("Please check the logs, your API keys, and internet connection.")

if __name__ == "__main__":
    main()