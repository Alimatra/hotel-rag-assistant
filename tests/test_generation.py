from hotel_rag.generation import build_prompt


def test_build_prompt_assembles_role_context_question_consigne_in_order():
    prompt = build_prompt(
        role="ROLE",
        context="## Piscine\n\nOuverte de 8h à 20h.",
        question="La piscine est-elle chauffée ?",
        consigne="CONSIGNE",
    )

    expected = (
        "ROLE\n\n## Piscine\n\nOuverte de 8h à 20h.\n\n"
        "Question d'un client : La piscine est-elle chauffée ?\nCONSIGNE"
    )
    assert prompt == expected
