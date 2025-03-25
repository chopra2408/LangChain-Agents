# NutriAgent - AI-Powered Nutrition and Diet Assistant

NutriAgent is an intelligent assistant that provides evidence-based nutrition and diet recommendations. It uses advanced AI models to research and generate diet plans tailored to user needs.

## Features

- **Personalized Diet Plans**: Generates structured diet plans based on user input, considering dietary preferences, allergies, and health goals.
- **Nutrition Research**: Conducts web searches on nutrition topics using trusted sources like NIH, Mayo Clinic, and Harvard Health.
- **AI-Powered Chat**: Answers general health, fitness, and dietary questions using GPT-4o.
- **Token-Efficient Responses**: Manages API calls efficiently to prevent token overuse.

## Technologies Used

- **Python 3**
- **LangChain** (for AI interactions)
- **OpenAI GPT-4o** (for text generation)
- **Tavily API** (for web search)
- **Logging & Error Handling** (for performance tracking)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/NutriAgent.git
   cd NutriAgent
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   Create a `.env` file and add your API keys:
   ```
   TAVILY_API_KEY=your_tavily_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

### Running the Application
```bash
python dietagent.py
```
### Commands
- `/search [query]` - Perform nutrition-related web searches.
- `/diet [requirements]` - Generate a personalized diet plan.
- `/bye` - Exit the program.

## Example
```bash
👤 You: /diet I need a 2000-calorie vegetarian meal plan for muscle gain.
🌿 NutriAgent: *[Generates a structured diet plan]*
```

## Future Enhancements
- Integration with food tracking apps.
- Expanded recipe generation with nutritional breakdowns.
- Voice-based interactions.

## License
This project is licensed under the MIT License.

## Author
[Nishant Chopra](https://github.com/yourusername)

