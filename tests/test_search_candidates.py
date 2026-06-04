"""Tests for relation-specific candidates extracted from search snippets."""

from __future__ import annotations

import unittest

from llmg.agent.search_candidates import candidate_matches, extract_search_candidates


class TestSearchCandidates(unittest.TestCase):
    def test_head_of_government_candidate(self) -> None:
        candidates = extract_search_candidates(
            "Who was the head of government in Portland, Oregon as of 2025-12-01?",
            "snippet: Mayor Neil Goldschmidt took office in 1972.",
        )
        self.assertEqual(candidates, ["Neil Goldschmidt"])

    def test_future_head_of_government_candidate_from_held_post_clause(self) -> None:
        candidates = extract_search_candidates(
            "Who was the head of government in Portland, Oregon as of 2026-01-01?",
            (
                "snippet: no Republican has served as mayor even on an interim basis since "
                "Connie McCready held the post from 1979 to 1980. "
                "Mayor Neil Goldschmidt took office in 1972."
            ),
        )
        self.assertEqual(candidates, ["Connie McCready"])

    def test_head_of_government_successor_candidate(self) -> None:
        candidates = extract_search_candidates(
            "Who is the head of government of Leipzig as of 2026-01-01?",
            "snippet: Wolfgang Tiefensee served until 2005. He was succeeded by fellow SPD politician Burkhard Jung, who was elected in January 2006.",
        )
        self.assertEqual(candidates, ["Burkhard Jung"])

    def test_diplomatic_relation_candidate(self) -> None:
        candidates = extract_search_candidates(
            "What country is Pakistan's diplomatic relation as_of 2025-12-01?",
            "snippet: Relations with Russia have improved since the end of the Cold War.",
        )
        self.assertEqual(candidates, ["Russia"])

    def test_head_coach_candidate(self) -> None:
        candidates = extract_search_candidates(
            "Who was the head coach of the Singapore men's national football team?",
            "snippet: Nazri Nasir was appointed as interim head coach of the Singapore national team.",
        )
        self.assertEqual(candidates, ["Nazri Nasir"])

    def test_replaced_head_coach_candidate(self) -> None:
        candidates = extract_search_candidates(
            "Who was the head coach of the New Zealand national rugby union team?",
            "snippet: Henry stepped down as coach and was replaced as head coach by his assistant Steve Hansen.",
        )
        self.assertEqual(candidates, ["Steve Hansen"])

    def test_year_specific_new_head_coach_candidate(self) -> None:
        candidates = extract_search_candidates(
            "Who was the head coach during the 2026 qualification campaign?",
            "snippet: During the 2026 FIFA World Cup qualification match, Singapore under new head coach Tsutomu Ogura played at home.",
        )
        self.assertEqual(candidates, ["Tsutomu Ogura"])

    def test_afc_head_coach_question_ignores_fifa_world_cup_candidate(self) -> None:
        candidates = extract_search_candidates(
            "Who was the head coach during the 2025 AFC Asian Cup qualification campaign?",
            (
                "snippet: During the 2026 FIFA World Cup qualification match, Singapore under new head coach Tsutomu Ogura played at home. "
                "On 7 March, 2019, Nazri Nasir was appointed as interim head coach of the Singapore national team."
            ),
        )
        self.assertEqual(candidates, ["Nazri Nasir"])

    def test_2026_afc_head_coach_question_allows_new_head_coach_candidate(self) -> None:
        candidates = extract_search_candidates(
            "Who was the head coach during the 2026 AFC Asian Cup qualification campaign?",
            (
                "snippet: During the 2026 FIFA World Cup qualification match, Singapore under new head coach Tsutomu Ogura played at home. "
                "On 7 March, 2019, Nazri Nasir was appointed as interim head coach of the Singapore national team."
            ),
        )
        self.assertEqual(candidates, ["Tsutomu Ogura", "Nazri Nasir"])

    def test_cofounded_company_candidate(self) -> None:
        candidates = extract_search_candidates(
            "Who was the company co-founded by Lawrence Joseph Ellison, as of 2026-02-01?",
            "snippet: Lawrence Joseph Ellison is an entrepreneur who co-founded the software company Oracle Corporation.",
        )
        self.assertEqual(candidates, ["Oracle Corporation"])

    def test_head_of_state_candidate(self) -> None:
        candidates = extract_search_candidates(
            "Who was the head of state of the Central African Republic as of 2025-12-01?",
            "snippet: Acting president since April 2016 is Faustin-Archange Touadéra who followed the interim government.",
        )
        self.assertEqual(candidates, ["Faustin-Archange Touadéra"])

    def test_spouse_prefers_westernized_name(self) -> None:
        candidates = extract_search_candidates(
            "Who was the spouse of William Oliver Stone as of 2025-12-01?",
            "snippet: Stone also has a daughter with his wife, Chong son Chong (Korean: x, alternately Westernized as Sun-jung Jung), whom he has been married to since 1996.",
        )
        self.assertEqual(candidates, ["Sun-jung Jung"])

    def test_first_husband_candidate_from_first_marriage(self) -> None:
        candidates = extract_search_candidates(
            "Who was Gloria Grahame's first husband, as of the 2026-01-01 snapshot?",
            "snippet: Grahame was married four times. Her first marriage was to actor Stanley Clements in August 1945.",
        )
        self.assertEqual(candidates, ["Stanley Clements"])

    def test_country_of_citizenship_from_naturalized_demonym(self) -> None:
        candidates = extract_search_candidates(
            "Target relation: country of citizenship. Return the full country of citizenship name/value.",
            "snippet: Doctorow became a British citizen by naturalisation on 12 August 2011.",
        )
        self.assertEqual(candidates, ["United Kingdom"])

    def test_brother_candidate_from_sons_list(self) -> None:
        candidates = extract_search_candidates(
            "Who was the son of Saul and one of the brothers of Ish-bosheth?",
            "snippet: Saul sired at least five sons (Jonathan, Abinadab, Malchishua, Ishvi and Ish-bosheth).",
        )
        self.assertEqual(candidates, ["Ishvi"])

    def test_position_held_candidate(self) -> None:
        candidates = extract_search_candidates(
            "What position did Ariel Sharon hold in the Israeli government as_of 2025-12-01?",
            "snippet: Sharon agreed to forfeit the post of defense minister but stayed in the cabinet as a minister without portfolio.",
        )
        self.assertEqual(candidates, ["minister without portfolio"])

    def test_prime_minister_position_candidate(self) -> None:
        candidates = extract_search_candidates(
            "Who was Ariel Sharon's position held in Israel as of 2026-01-01?",
            "snippet: Ariel Sharon was an Israeli politician who served as the prime minister of Israel from March 2001 until April 2006.",
        )
        self.assertEqual(candidates, ["Prime Minister of Israel"])

    def test_year_specific_position_candidate_from_became_clause(self) -> None:
        candidates = extract_search_candidates(
            "In which position was Harold Wilson appointed in 1947 as part of the Attlee government?",
            "snippet: Wilson was appointed to the Attlee government as a Parliamentary secretary; he became Secretary for Overseas Trade in 1947.",
        )
        self.assertEqual(candidates, ["Secretary for Overseas Trade"])

    def test_record_label_released_through_candidate(self) -> None:
        candidates = extract_search_candidates(
            "Which record label did Ed Sheeran launch in September 2023?",
            'snippet: Autumn Variations was released on 29 September 2023. It is his first studio album to be released through Gingerbread Man Records.',
        )
        self.assertEqual(candidates, ["Gingerbread Man Records"])

    def test_record_label_released_on_candidate(self) -> None:
        candidates = extract_search_candidates(
            'Which record label was Donna Summer signed to for the release of her 1983 album "She Works Hard for the Money"?',
            'snippet: The result was the album "She Works Hard for the Money", produced by Michael Omartian, and released in 1983 on Mercury Records.',
        )
        self.assertEqual(candidates, ["Mercury Records"])

    def test_current_ipl_team_candidate(self) -> None:
        candidates = extract_search_candidates(
            "Which cricket team did Suryakumar Yadav play for in the Indian Premier League?",
            "snippet: Yadav plays for Mumbai Indians and has previously played for Kolkata Knight Riders in the Indian Premier League.",
        )
        self.assertEqual(candidates, ["Mumbai Indians"])

    def test_domestic_cricket_team_candidate_from_selected_clause(self) -> None:
        candidates = extract_search_candidates(
            "Which cricket team did Suryakumar Yadav represent in domestic cricket as of August 2024?",
            "snippet: Yadav started playing club cricket in Mumbai before he was selected for the Mumbai cricket team.",
        )
        self.assertEqual(candidates, ["Mumbai cricket team"])

    def test_radio_station_candidate_expands_surface(self) -> None:
        candidates = extract_search_candidates(
            "What radio station is owned by the University of California San Diego?",
            "snippet: student media include Triton Television and the KSDT radio station.",
        )
        self.assertEqual(candidates, ["KSDT Radio"])

    def test_baseball_facility_candidate(self) -> None:
        candidates = extract_search_candidates(
            "What facility, located on the campus of the University of California San Diego, is known for hosting collegiate baseball games?",
            "snippet: UCSD's teams compete at the university's RIMAC facility, Triton Ballpark, and LionTree Arena.",
        )
        self.assertEqual(candidates, ["Triton Ballpark"])

    def test_release_studio_candidate(self) -> None:
        candidates = extract_search_candidates(
            'Which studio was responsible for the release of "Sonic the Hedgehog 3"?',
            'snippet: "Sonic the Hedgehog 3" was released by Paramount Pictures in the United States on December 20.',
        )
        self.assertEqual(candidates, ["Paramount Pictures"])

    def test_member_of_sports_team_candidate_from_brief_spell(self) -> None:
        candidates = extract_search_candidates(
            "Target relation: member of sports team. Which team did the player join?",
            "snippet: After a brief spell with Debreceni EAC, a club of the Hungarian Nemzeti Bajnokság I/A, Costa returned to FC Barcelona B.",
        )
        self.assertEqual(candidates, ["Debreceni EAC"])

    def test_basketball_team_candidate_from_joined_clause(self) -> None:
        candidates = extract_search_candidates(
            "Which professional basketball team did Greg Ostertag join after leaving the Utah Jazz?",
            "snippet: After nine seasons in Utah, he joined the Sacramento Kings.",
        )
        self.assertEqual(candidates, ["Sacramento Kings"])

    def test_basketball_team_candidate_from_final_seasons(self) -> None:
        candidates = extract_search_candidates(
            "Which professional basketball team did Dell Curry play for during his final three seasons in the NBA?",
            "snippet: Curry played one season for the Milwaukee Bucks before playing his final three seasons in the NBA for the Toronto Raptors.",
        )
        self.assertEqual(candidates, ["Toronto Raptors"])

    def test_football_team_candidate_from_traded_clause(self) -> None:
        candidates = extract_search_candidates(
            "Which professional football team did Brett Favre join after being traded in 2008?",
            "snippet: Favre was traded in 2008 to the New York Jets, where he played one year.",
        )
        self.assertEqual(candidates, ["New York Jets"])

    def test_basketball_team_candidate_from_dated_traded_clause(self) -> None:
        candidates = extract_search_candidates(
            "Which professional basketball team did Mike Scott join after being traded on February 6, 2019?",
            "snippet: On February 6, 2019, Scott was traded, along with Tobias Harris, to the Philadelphia 76ers in exchange for Wilson Chandler.",
        )
        self.assertEqual(candidates, ["Philadelphia 76ers"])

    def test_basketball_team_candidate_from_midseason_trade_parenthetical(self) -> None:
        candidates = extract_search_candidates(
            "Which professional basketball team did James Donaldson play for during the 1991–92 season?",
            (
                "snippet: After brief stints with the New York Knicks "
                "(traded midway through 1991–92 for Brian Quinnett) and Utah Jazz "
                "(49 games in two seasons combined) in the early 1990s."
            ),
        )
        self.assertEqual(candidates, ["Utah Jazz"])

    def test_basketball_team_candidate_ignores_wrong_year_trade(self) -> None:
        candidates = extract_search_candidates(
            "Which professional basketball team did Mike Scott join in 2018?",
            "snippet: On February 23, 2017, Scott was traded, along with rights and cash considerations, to the Phoenix Suns in exchange for a pick.",
        )
        self.assertEqual(candidates, [])

    def test_basketball_team_candidate_from_dated_signed_clause(self) -> None:
        candidates = extract_search_candidates(
            "Which professional basketball team did Mike Scott join in 2018?",
            "snippet: On July 9, 2018, Scott signed with the Los Angeles Clippers.",
        )
        self.assertEqual(candidates, ["Los Angeles Clippers"])

    def test_basketball_team_candidate_from_signed_announcement(self) -> None:
        candidates = extract_search_candidates(
            "Which professional basketball team did Lorenzo Brown join in August 2019?",
            "snippet: On August 3, 2019, KK Crvena zvezda announced that they had signed Brown.",
        )
        self.assertEqual(candidates, ["KK Crvena zvezda"])

    def test_candidate_matches_normalizes_punctuation(self) -> None:
        self.assertTrue(candidate_matches("the Network", "The Network"))
        self.assertFalse(candidate_matches("Sam Adams", "Neil Goldschmidt"))


if __name__ == "__main__":
    unittest.main()
