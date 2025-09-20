# AI-Powered Customer Support Ticket Classifier

An intelligent application that automatically analyzes and categorizes incoming customer support tickets. The system assigns priority levels (Urgent, High, Medium, Low) and routes tickets to the appropriate departments (Tech Support, Billing, Sales).

## Features

- **Automatic Priority Classification**: Assigns priority levels based on ticket content
- **Department Routing**: Routes tickets to Tech Support, Billing, or Sales departments
- **RESTful API**: FastAPI-based API for easy integration
- **Machine Learning**: Uses scikit-learn for text classification
- **Confidence Scoring**: Provides confidence scores for predictions

## Tech Stack

- **Python 3.8+**: Core programming language
- **FastAPI**: Modern web framework for building APIs
- **scikit-learn**: Machine learning library for classification
- **Pandas**: Data manipulation and analysis
- **TF-IDF Vectorization**: Text feature extraction
- **Logistic Regression**: Classification algorithm

## Project Structure

```
├── app/
│   ├── models/
│   │   ├── classifier.py      # ML classifier implementation
│   │   └── schemas.py         # Pydantic models for API
│   ├── api/
│   └── main.py               # FastAPI application
├── data/
│   └── generate_dataset.py   # Sample dataset generation
├── tests/
│   ├── test_classifier.py    # Classifier tests
│   └── test_api.py          # API tests
├── train.py                 # Model training script
└── requirements.txt         # Dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/BoonBoonBoonBoon/AI-Powered-Customer-Support-Ticket-Classifier.git
cd AI-Powered-Customer-Support-Ticket-Classifier
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Train the Model

Generate sample data and train the classifier:

```bash
python train.py
```

This will:
- Generate a sample dataset with 1000 tickets
- Train both priority and department classifiers
- Save the trained models to the `models/` directory

### 2. Start the API Server

```bash
python -m app.main
```

Or with uvicorn directly:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### 3. Test the API

You can test the API using the interactive documentation at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

Or use curl:

```bash
curl -X POST "http://localhost:8000/classify" \
     -H "Content-Type: application/json" \
     -d '{
       "title": "Server is down",
       "description": "The main server is not responding and all users are affected",
       "customer_email": "customer@example.com"
     }'
```

## API Endpoints

### POST /classify
Classify a customer support ticket.

**Request Body:**
```json
{
  "title": "string",
  "description": "string",
  "customer_email": "string (optional)"
}
```

**Response:**
```json
{
  "title": "string",
  "description": "string",
  "predicted_priority": "Urgent|High|Medium|Low",
  "predicted_department": "Tech Support|Billing|Sales",
  "priority_confidence": 0.95,
  "department_confidence": 0.87,
  "customer_email": "string"
}
```

### GET /health
Check the health status of the API and model.

### GET /model/status
Get the current status of the trained models.

## Training with Custom Data

To train with your own dataset, prepare a CSV file with the following columns:
- `title`: Ticket title
- `description`: Ticket description
- `priority`: Priority level (Urgent, High, Medium, Low)
- `department`: Department (Tech Support, Billing, Sales)

Then run:

```bash
python train.py --data path/to/your/data.csv
```

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run tests with coverage:

```bash
pytest tests/ --cov=app --cov-report=html
```

## Example Classifications

The system can classify various types of tickets:

**Tech Support Tickets:**
- Server outages → Urgent priority
- Login issues → High priority
- Feature requests → Low priority

**Billing Tickets:**
- Payment failures → High priority
- Invoice questions → Medium priority
- Address changes → Low priority

**Sales Tickets:**
- Demo requests → High priority
- Pricing inquiries → Medium priority
- Feature comparisons → Low priority

## Configuration

The application uses sensible defaults but can be configured through environment variables or by modifying the source code. Key configuration options include:

- Model parameters (in `app/models/classifier.py`)
- API settings (in `app/main.py`)
- Training parameters (in `train.py`)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.