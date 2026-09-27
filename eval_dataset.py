"""
Evaluation Dataset
===================
A hand-labeled test set: each ticket has a "ground truth" category and
urgency that a human (you) decided is correct. This lets us measure the
classifier's accuracy objectively, instead of eyeballing a few examples.

Design notes for this 20-ticket set:
- Every category appears at least twice, so a weak category doesn't hide
  behind one lucky/unlucky prediction.
- Tickets reference the expanded 10-order mock database (orders.json),
  covering varied statuses (Processing, Return Initiated, etc.) so the
  tool-calling step also gets exercised across more real scenarios.
- A couple of deliberately ambiguous/tricky tickets are included (e.g.
  one that could plausibly be two categories) - these are the ones most
  likely to reveal real classifier weaknesses, not just confirm it works
  on easy cases.
"""

from classifier import Category, Urgency

EVAL_DATASET = [
    # --- Order Status ---
    {
        "ticket": "Where can I track my order ORD12348?",
        "expected_category": Category.ORDER_STATUS,
        "expected_urgency": Urgency.LOW,
    },
    {
        "ticket": "Has my order ORD12350 been shipped yet? It's been a few days since I ordered.",
        "expected_category": Category.ORDER_STATUS,
        "expected_urgency": Urgency.LOW,
    },

    # --- Delivery Problem ---
    {
        "ticket": "My order ORD12345 hasn't arrived yet, it's already 3 days late",
        "expected_category": Category.DELIVERY_PROBLEM,
        "expected_urgency": Urgency.MEDIUM,
    },
    {
        "ticket": "Order ORD12353 was supposed to arrive today but tracking hasn't updated in 2 days",
        "expected_category": Category.DELIVERY_PROBLEM,
        "expected_urgency": Urgency.MEDIUM,
    },

    # --- Payment ---
    {
        "ticket": "Paid Rs 499 for a case but order ORD12346 shows Payment Failed, money got deducted",
        "expected_category": Category.PAYMENT,
        "expected_urgency": Urgency.HIGH,
    },
    {
        "ticket": "My card was charged twice for the same order, please check",
        "expected_category": Category.PAYMENT,
        "expected_urgency": Urgency.HIGH,
    },
    {
        "ticket": "What payment methods do you accept?",
        "expected_category": Category.GENERAL,
        "expected_urgency": Urgency.LOW,
    },

    # --- Returns/Refund ---
    {
        "ticket": "Can you tell me the return window for electronics?",
        "expected_category": Category.RETURNS_REFUND,
        "expected_urgency": Urgency.LOW,
    },
    {
        "ticket": "Received a damaged headphone set, the box was crushed",
        "expected_category": Category.RETURNS_REFUND,
        "expected_urgency": Urgency.HIGH,
    },
    {
        "ticket": "I want a refund for order ORD99999 immediately, this is unacceptable",
        "expected_category": Category.RETURNS_REFUND,
        "expected_urgency": Urgency.HIGH,
    },
    {
        "ticket": "It's been 6 days since I initiated a return for ORD12352, when will I get my refund?",
        "expected_category": Category.RETURNS_REFUND,
        "expected_urgency": Urgency.MEDIUM,
    },

    # --- Cancellation ---
    {
        "ticket": "I want to cancel order ORD12350 before it ships",
        "expected_category": Category.CANCELLATION,
        "expected_urgency": Urgency.MEDIUM,
    },
    {
        "ticket": "Can I still cancel ORD12353? I ordered the wrong item by mistake",
        "expected_category": Category.CANCELLATION,
        "expected_urgency": Urgency.MEDIUM,
    },

    # --- Account/Address ---
    {
        "ticket": "URGENT - I think someone accessed my account without my permission and changed my password",
        "expected_category": Category.ACCOUNT_ADDRESS,
        "expected_urgency": Urgency.CRITICAL,
    },
    {
        "ticket": "Can I change my delivery address for order ORD12348? It hasn't shipped yet",
        "expected_category": Category.ACCOUNT_ADDRESS,
        "expected_urgency": Urgency.MEDIUM,
    },
    {
        "ticket": "I forgot my account password and the reset email isn't arriving",
        "expected_category": Category.ACCOUNT_ADDRESS,
        "expected_urgency": Urgency.MEDIUM,
    },

    # --- Product Inquiry ---
    {
        "ticket": "Does the wireless headphone support Bluetooth 5.0?",
        "expected_category": Category.PRODUCT_INQUIRY,
        "expected_urgency": Urgency.LOW,
    },
    {
        "ticket": "Is the smart watch (ORD12350) water resistant?",
        "expected_category": Category.PRODUCT_INQUIRY,
        "expected_urgency": Urgency.LOW,
    },

    # --- General ---
    {
        "ticket": "Do you have a physical store I can visit in Ahmedabad?",
        "expected_category": Category.GENERAL,
        "expected_urgency": Urgency.LOW,
    },

    # --- Deliberately tricky/ambiguous ticket ---
    {
        "ticket": "I returned my yoga mat (ORD12352) a week ago and haven't received my money back, "
                  "and now I also want to cancel my other pending order ORD12350",
        "expected_category": Category.RETURNS_REFUND,  # primary intent is the overdue refund
        "expected_urgency": Urgency.MEDIUM,
    },
]