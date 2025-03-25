import os
import time
import json
import logging
import re
from typing import Dict, List, Optional, Union
from tavily import TavilyClient
from dotenv import load_dotenv
from langchain.adapters.openai import convert_openai_messages
from langchain_openai import ChatOpenAI
from tiktoken import get_encoding
from langchain_core.messages import HumanMessage, SystemMessage

# Configure logging for detailed debugging and performance tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class MessageHistory:
    """Class for managing conversation history"""
    def __init__(self, max_history: int = 10):
        self.history = []
        self.max_history = max_history
    
    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        # Trim history if it gets too long
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
            temperature=0.4,
            max_tokens=2000
        )
        self.encoding = get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Return the token count for a given text using the provided encoding."""
        return len(self.encoding.encode(text))
    
    def safe_api_invoke(self, messages, max_retries=3, retry_delay=2):
        """Invoke the API safely with retries on rate limit errors."""
        for attempt in range(max_retries):
            try:
                # Use invoke() instead of chat()
                response = self.chat_model.invoke(messages)
                return response
            except Exception as e:
                logging.warning(f"API call failed on attempt {attempt+1}/{max_retries}: {e}")
                time.sleep(retry_delay * (attempt + 1))
                if attempt == max_retries - 1:
                    raise Exception(f"Exceeded maximum retry attempts for API invocation: {e}")
    
    def search_for_information(self, query: str) -> Dict:
        """Perform web searches for nutrition information"""
        logging.info(f"Researching nutrition query: {query}")
        
        try:
            # Execute a balanced search within token limits
            search_results = self.tavily_client.search(
                query=query, 
                max_results=7,
                search_depth="advanced",
                include_answer=True,
                include_raw_content=False,
                time_range="month",
                include_domains=[
                    "nih.gov", "mayoclinic.org", "health.harvard.edu", 
                    "medicalnewstoday.com", "healthline.com", "webmd.com", 
                    "ncbi.nlm.nih.gov", "hopkinsmedicine.org"
                ]
            )['results']
            logging.info(f"Found {len(search_results)} general sources.")

            # Perform a targeted search for scientific perspective
            scientific_query = f"scientific research summary {query}"
            scientific_results = self.tavily_client.search(
                query=scientific_query,
                max_results=3,
                search_depth="advanced",
                include_domains=["ncbi.nlm.nih.gov", "pubmed.ncbi.nlm.nih.gov"]
            )['results']
            logging.info(f"Found {len(scientific_results)} scientific sources.")

            # Build an efficient research summary (minimizing token usage)
            research_summary = {
                "general_results": [
                    {"url": r["url"], "title": r["title"], "content": r.get("snippet", "")}
                    for r in search_results
                ],
                "scientific_perspective": [
                    {"url": r["url"], "title": r["title"], "content": r.get("snippet", "")}
                    for r in scientific_results
                ]
            }
            
            return research_summary
        except Exception as e:
            logging.error(f"Error during web search: {e}")
            return {
                "general_results": [],
                "scientific_perspective": [],
                "error": str(e)
            }
    
    def generate_nutrition_response(self, query: str, research_summary: Dict) -> str:
        """Generate a comprehensive nutrition response based on search results"""
        system_message = (
            "You are 'FitMentor', an expert in nutrition and health sciences with credentials in dietetics "
            "and exercise physiology. Provide scientifically accurate, detailed, and actionable responses.\n\n"
            "CONTENT REQUIREMENTS:\n"
            "- Evidence-based information with specific detailed data points\n"
            "- Practical recommendations (including dosages and timing where applicable)\n"
            "- Discussion of risks, contraindications, and population-specific effects\n"
            "- Quality of evidence (strong, moderate, preliminary) and mechanisms when relevant\n\n"
            "RESPONSE STRUCTURE:\n"
            "1. Executive Summary in 4-5 sentences highlighting the full response about the user query\n"
            "2. Main Sections with Subsections and subsections should be 4-5 points\n"
            "3. Practical Recommendations should have 4-5 points\n"
            "4. Risk Assessment\n"
            "5. Special Considerations\n"
            "6. Related Questions: always add at least 5 related questions\n"
            "7. Key Takeaways\n"
            "8. References\n\n"
            "FORMATTING:\n"
            "- Use markdown with clear headings (H2, H3) and bullet points\n"
            "- **Bold** important warnings or critical points\n\n"
            "AIM FOR SUPERIOR QUALITY: Provide large content but keep it specific and actionable advice."
        )
        
        user_message = (
            f"RESEARCH DATA: {json.dumps(research_summary)}\n\n"
            f"QUERY: \"{query}\"\n\n"
            "INSTRUCTIONS:\n"
            "1. Create a comprehensive response with detailed data points, dosages, and actionable advice.\n"
            "2. Address risks and benefits with scientific context and quality of evidence.\n"
            "3. Format the answer for clarity using markdown, with well-organized sections.\n"
            "4. Provide credible references at the end.\n\n"
            "IMPORTANT: Focus on delivering quality and specificity, surpassing typical AI outputs."
        )
        
        # Format messages for LangChain
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_message)
        ]
        
        # Token management
        total_input_tokens = self.count_tokens(system_message) + self.count_tokens(user_message)
        max_output_tokens = 2000
        max_context_tokens = 128000
        
        if total_input_tokens + max_output_tokens > max_context_tokens:
            logging.warning("Token limit exceeded, truncating research data")
            # Truncate research data if needed
            truncated_research = self._truncate_research(research_summary, max_context_tokens - max_output_tokens - self.count_tokens(system_message))
            user_message = (
                f"RESEARCH DATA: {json.dumps(truncated_research)}\n\n"
                f"QUERY: \"{query}\"\n\n"
                "INSTRUCTIONS: Create a concise but comprehensive response using the available data."
            )
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=user_message)
            ]
        
        logging.info("Generating nutrition response...")
        start_time = time.time()
        
        # Use safe_api_invoke with messages (using invoke() under the hood)
        response = self.safe_api_invoke(messages)
        
        elapsed = time.time() - start_time
        logging.info(f"Response generated in {elapsed:.2f} seconds.")
        
        return response.content
    
    def _truncate_research(self, research_summary: Dict, max_tokens: int) -> Dict:
        """Truncate research data to fit within token limits"""
        truncated = {
            "general_results": [],
            "scientific_perspective": []
        }
        
        # Add scientific results first (prioritize)
        for item in research_summary["scientific_perspective"]:
            item_json = json.dumps(item)
            if self.count_tokens(json.dumps(truncated)) + self.count_tokens(item_json) < max_tokens:
                truncated["scientific_perspective"].append(item)
            else:
                break
        
        # Add general results with remaining token budget
        for item in research_summary["general_results"]:
            item_json = json.dumps(item)
            if self.count_tokens(json.dumps(truncated)) + self.count_tokens(item_json) < max_tokens:
                truncated["general_results"].append(item)
            else:
                break
                
        return truncated

class DietPlanAgent:
    """Agent for creating personalized diet plans"""
    def __init__(self, model: str = "gpt-4o"):
        self.chat_model = ChatOpenAI(
            model=model,
            temperature=0.4,
        )
        self.encoding = get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Return the token count for a given text using the provided encoding."""
        return len(self.encoding.encode(text))
    
    def safe_api_invoke(self, messages, max_retries=3, retry_delay=2):
        """Invoke the API safely with retries on rate limit errors."""
        for attempt in range(max_retries):
            try:
                response = self.chat_model.invoke(messages)
                return response
            except Exception as e:
                logging.warning(f"API call failed on attempt {attempt+1}/{max_retries}: {e}")
                time.sleep(retry_delay * (attempt + 1))
                if attempt == max_retries - 1:
                    raise Exception(f"Exceeded maximum retry attempts for API invocation: {e}")
    
    def parse_user_requirements(self, query: str) -> Dict:
        """Extract key dietary requirements from user query"""
        system_prompt = """
        You are a dietary analysis assistant. Extract key dietary information from the user's query.
        Return ONLY a JSON object with the following keys (use null if information is not provided):
        
        {
          "diet_type": null,           # e.g., "keto", "vegetarian", "paleo", "mediterranean", "standard"
          "calorie_target": null,      # e.g., 1800, 2000, 2500
          "protein_target": null,      # in grams or as percentage
          "carb_target": null,         # in grams or as percentage
          "fat_target": null,          # in grams or as percentage
          "meals_per_day": 3,          # default to 3 if not specified
          "allergies": [],             # list of allergies
          "restrictions": [],          # dietary restrictions
          "preferences": [],           # food preferences
          "health_goals": [],          # e.g., "weight loss", "muscle gain", "improve energy"
          "timeframe": "daily",        # "daily", "weekly"
          "days": null                 # number of days the plan should cover (if weekly and days < 7, then set to 7)
        }
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Extract dietary information from this query: {query}")
        ]
        
        response = self.safe_api_invoke(messages)
        
        try:
            # Extract JSON from the response
            json_str = response.content
            json_str = re.sub(r'```json', '', json_str)
            json_str = re.sub(r'```', '', json_str)
            requirements = json.loads(json_str.strip())
            
            # Adjust days based on timeframe:
            if requirements.get("timeframe") == "weekly":
                # If days is not specified or less than 7, set to 7.
                if not requirements.get("days") or requirements.get("days") < 7:
                    requirements["days"] = 7
            else:
                # For daily plans, if days is not specified or less than 1, set to 1.
                if not requirements.get("days") or requirements.get("days") < 1:
                    requirements["days"] = 1
                    
            return requirements
        except Exception as e:
            logging.error(f"Error parsing dietary requirements: {e}")
            # Return default values if parsing fails
            return {
                "diet_type": "standard",
                "calorie_target": None,
                "protein_target": None,
                "carb_target": None,
                "fat_target": None,
                "meals_per_day": 3,
                "allergies": [],
                "restrictions": [],
                "preferences": [],
                "health_goals": [],
                "timeframe": "daily",
                "days": 1
            }
    
    def generate_diet_plan(self, query: str) -> Dict:
        """Generate a structured diet plan with detailed meals and instructions"""
        logging.info(f"Creating diet plan for: {query}")
        
        # Parse user requirements
        requirements = self.parse_user_requirements(query)
        days = requirements.get("days", 1)
        meals = requirements.get("meals_per_day", 3)
        
        # Build system prompt
        system_prompt = """
        You are 'NutriChef', an expert nutritionist and culinary professional specialized in creating personalized meal plans.
        
        Your task is to create a detailed, nutritionally balanced diet plan in JSON format. The plan should strictly follow this structure:
        
        {
          "diet_plan": {
            "plan_name": "Name of the personalized diet plan",
            "plan_description": "Brief overview of the plan",
            "dietary_approach": "Diet type (keto, Mediterranean, etc.)",
            "nutritional_targets": {
              "calories": 0,
              "protein": "0g (00%)",
              "carbs": "0g (00%)",
              "fat": "0g (00%)"
            },
            "days": [
              {
                "day_number": 1,
                "meals": [
                  {
                    "meal_name": "Breakfast",
                    "meal_time": "7:00 AM - 9:00 AM",
                    "recipes": [
                      {
                        "recipe_name": "Name of dish",
                        "preparation_time": "00 minutes",
                        "difficulty": "Easy/Medium/Hard",
                        "ingredients": [
                          {"name": "Ingredient 1", "amount": "100g", "calories": 000, "protein": "00g", "carbs": "00g", "fat": "00g"},
                          {"name": "Ingredient 2", "amount": "1 tbsp", "calories": 000, "protein": "00g", "carbs": "00g", "fat": "00g"}
                        ],
                        "instructions": [
                          "Step 1 with detailed description",
                          "Step 2 with detailed description"
                        ],
                        "nutritional_info": {
                          "calories": 000,
                          "protein": "00g",
                          "carbs": "00g",
                          "fat": "00g"
                        },
                        "tips": "Optional preparation or nutrition tips"
                      }
                    ],
                    "total_meal_nutrition": {
                      "calories": 000,
                      "protein": "00g",
                      "carbs": "00g",
                      "fat": "00g"
                    }
                  },
                  {
                    "meal_name": "Lunch",
                    "meal_time": "12:00 PM - 2:00 PM",
                    "recipes": [...]
                  },
                  {
                    "meal_name": "Dinner",
                    "meal_time": "6:00 PM - 8:00 PM",
                    "recipes": [...]
                  }
                ],
                "daily_totals": {
                  "calories": 0000,
                  "protein": "000g",
                  "carbs": "000g",
                  "fat": "000g"
                }
              }
              // Repeat for each day
            ],
            "hydration": {
              "recommended_water": "0L per day",
              "timing": "Distribution throughout the day"
            },
            "notes": [
              "Important consideration 1",
              "Important consideration 2"
            ],
            "scientific_rationale": "Brief explanation of the scientific basis for this diet plan"
          }
        }
        
        IMPORTANT GUIDELINES:
        - Generate a complete diet plan for exactly {days} days.
        - Include {meals} meals per day with detailed recipes, preparation instructions (at least 3-5 steps per recipe), and exact nutritional information.
        - Ensure meals are practical to prepare using commonly available ingredients and meet any dietary restrictions and preferences.
        - Ensure daily totals are accurate calculations from all meals.
        """
        
        # Build user prompt
        user_prompt = f"""
        Create a detailed personalized diet plan based on the following information:
        
        USER QUERY: {query}
        
        EXTRACTED REQUIREMENTS: {json.dumps(requirements, indent=2)}
        
        Generate a complete diet plan in the structured JSON format for {days} days. Include {meals} meals per day.
        Provide all calculations accurately and ensure the plan is practical to prepare.
        Return ONLY the JSON without any additional text or markdown.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        logging.info("Generating diet plan...")
        start_time = time.time()
        response = self.safe_api_invoke(messages)
        elapsed = time.time() - start_time
        logging.info(f"Diet plan generated in {elapsed:.2f} seconds")
        
        try:
            json_str = response.content
            json_str = re.sub(r'```json', '', json_str)
            json_str = re.sub(r'```', '', json_str)
            diet_plan = json.loads(json_str.strip())
            
            timestamp = int(time.time())
            filename = f"diet_plan_{timestamp}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(diet_plan, f, indent=2)
            logging.info(f"Diet plan saved to {filename}")
            
            return diet_plan
        except Exception as e:
            logging.error(f"Error parsing diet plan: {e}")
            return {
                "error": str(e),
                "partial_response": response.content
            }

# In ParentAI, modify the /diet command to call generate_diet_plan (which returns JSON)
class ParentAI:
    """Parent AI that handles conversation routing and general queries"""
    def __init__(self):
        load_dotenv()
        self.tavily_api_key = os.environ.get('TAVILY_API_KEY')
        if not self.tavily_api_key:
            raise ValueError("TAVILY_API_KEY not found in environment variables.")
        
        self.nutrition_agent = NutritionAgent(api_key=self.tavily_api_key)
        self.diet_plan_agent = DietPlanAgent()
        self.conversation_history = MessageHistory()
        self.chat_model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1000
        )
    
    def is_nutrition_query(self, query: str) -> bool:
        system_prompt = (
            "You are a classifier that determines if a user query is related to nutrition, "
            "health, diet, supplements, fitness, or wellness. Respond with only 'YES' or 'NO'."
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Query: {query}\nIs this related to nutrition, health, diet, supplements, fitness, or wellness? Answer YES or NO only.")
        ]
        response = self.chat_model.invoke(messages)
        result = response.content.strip().upper()
        return "YES" in result
    
    def is_diet_plan_request(self, query: str) -> bool:
        system_prompt = (
            "You are a classifier that determines if a user query is specifically requesting a diet plan, "
            "meal plan, or food regimen. The query must be asking for an actual plan, not just nutrition advice. "
            "Respond with only 'YES' or 'NO'."
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Query: {query}\nIs this requesting a diet plan, meal plan, or food regimen? Answer YES or NO only.")
        ]
        response = self.chat_model.invoke(messages)
        result = response.content.strip().upper()
        return "YES" in result
    
    def handle_general_query(self, query: str) -> str:
        system_prompt = (
            "You are a helpful and friendly assistant. Provide conversational responses "
            "to general queries. Keep responses concise and natural."
        )
        history = self.conversation_history.get_history()
        messages = [SystemMessage(content=system_prompt)]
        for msg in history[-5:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(SystemMessage(content=f"Assistant: {msg['content']}"))
        messages.append(HumanMessage(content=query))
        response = self.chat_model.invoke(messages)
        return response.content
    
    def process_query(self, query: str) -> str:
        if query.lower() == '/bye':
            return "Goodbye! Shutting down..."
        
        if query.lower().startswith('/search '):
            search_query = query[8:].strip()
            if not search_query:
                return "Please provide a search query after /search."
            try:
                research_data = self.nutrition_agent.search_for_information(search_query)
                response = self.nutrition_agent.generate_nutrition_response(search_query, research_data)
                timestamp = int(time.time())
                filename = f"search_response_{timestamp}.md"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(response)
                logging.info(f"Search response saved to {filename}")
                return response
            except Exception as e:
                logging.error(f"Error in search processing: {e}")
                return f"I encountered an error while searching: {str(e)}. Please try again."
        
        if query.lower().startswith('/diet '):
            diet_query = query[6:].strip()
            if not diet_query:
                return "Please provide details for your diet plan request after /diet."
            try:
                # Call generate_diet_plan so that JSON is returned
                response = self.diet_plan_agent.generate_diet_plan(diet_query)
                return json.dumps(response, indent=2)
            except Exception as e:
                logging.error(f"Error in diet plan processing: {e}")
                return f"I encountered an error while creating your diet plan: {str(e)}. Please try again with more specific requirements."
        
        self.conversation_history.add_message("user", query)
        if self.is_diet_plan_request(query):
            logging.info("Detected diet plan request, routing to DietPlanAgent")
            try:
                response = self.diet_plan_agent.generate_diet_plan(query)
                response = json.dumps(response, indent=2)
            except Exception as e:
                logging.error(f"Error in diet plan processing: {e}")
                response = f"I encountered an error while creating your diet plan: {str(e)}. Please try again with more specific requirements."
        elif self.is_nutrition_query(query):
            logging.info("Detected nutrition query, routing to NutritionAgent")
            try:
                research_data = self.nutrition_agent.search_for_information(query)
                response = self.nutrition_agent.generate_nutrition_response(query, research_data)
                timestamp = int(time.time())
                filename = f"nutrition_response_{timestamp}.md"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(response)
                logging.info(f"Nutrition response saved to {filename}")
            except Exception as e:
                logging.error(f"Error in nutrition processing: {e}")
                response = f"I encountered an error while researching nutrition information: {str(e)}. Please try again with a more specific query."
        else:
            logging.info("Detected general query, handling with ParentAI")
            response = self.handle_general_query(query)
        
        self.conversation_history.add_message("assistant", response)
        return response

def main():
    """Main function to run the conversational agent"""
    print("🌿 Welcome to NutriAgent - Your AI Nutrition and Health Assistant")
    print("Type '/bye' to exit. Use '/search [query]' for direct web search. Use '/diet [details]' for personalized diet plans.")
    
    try:
        parent_ai = ParentAI()
        running = True
        
        while running:
            user_input = input("\n👤 You: ").strip()
            print("\n🌿 Processing...")
            response = parent_ai.process_query(user_input)
            print(f"\n🌿 NutriAgent: {response}")
            
            if user_input.lower() == '/bye':
                running = False
                print("\n🌿 Thank you for using NutriAgent. Have a healthy day!")
                
    except KeyboardInterrupt:
        print("\n\n🌿 Session ended by user. Have a healthy day!")
    except Exception as e:
        logging.error(f"Critical error: {e}")
        print(f"\n🌿 An error occurred: {e}")
        print("Please check your API keys and internet connection.")

if __name__ == "__main__":
    main()