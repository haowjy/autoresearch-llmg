"""Unit tests for agent read_file pagination and grep_file path validation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llmg.agent.sandbox import AgentSandbox
from llmg.agent.state import AgentEpisodeState
from llmg.agent.tools import READ_FILE_DEFAULT_LIMIT, make_tool_functions
from llmg.memory.fs_store import FsStore


class TestReadFilePagination(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self._tmp.name)
        articles = self.workspace / "articles"
        articles.mkdir(parents=True)
        self.body = "HEAD" + ("x" * 5000) + "TAIL_MARKER" + ("y" * 500)
        (articles / "sample.md").write_text(self.body, encoding="utf-8")
        self.sandbox = AgentSandbox(self.workspace)
        self.fs = FsStore(self.workspace)
        self.episode = AgentEpisodeState()
        _, self.tools = make_tool_functions(
            sandbox=self.sandbox,
            fs_store=self.fs,
            episode=self.episode,
            toolset="shell",
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_default_first_slice(self) -> None:
        out = self.tools["read_file"](path="articles/sample.md")
        self.assertEqual(out, self.body[:READ_FILE_DEFAULT_LIMIT])
        self.assertIn("HEAD", out)
        self.assertNotIn("TAIL_MARKER", out)

    def test_offset_limit_slice(self) -> None:
        off = self.body.index("TAIL_MARKER")
        out = self.tools["read_file"](path="articles/sample.md", offset=off, limit=50)
        self.assertTrue(out.startswith("TAIL_MARKER"))
        self.assertLessEqual(len(out), 50)

    def test_repeated_exact_slice_is_skipped(self) -> None:
        first = self.tools["read_file"](path="articles/sample.md", offset=0, limit=50)
        second = self.tools["read_file"](path="articles/sample.md", offset=0, limit=50)
        self.assertEqual(first, self.body[:50])
        self.assertIn("repeat read_file skipped", second)
        self.assertIn("submit_answer", second)

    def test_different_slice_after_repeat_is_allowed(self) -> None:
        first = self.tools["read_file"](path="articles/sample.md", offset=0, limit=50)
        second = self.tools["read_file"](path="articles/sample.md", offset=50, limit=50)
        self.assertEqual(first, self.body[:50])
        self.assertEqual(second, self.body[50:100])

    def test_offset_past_end(self) -> None:
        out = self.tools["read_file"](path="articles/sample.md", offset=len(self.body) + 10)
        self.assertIn("empty slice", out)

    def test_path_escape_rejected(self) -> None:
        out = self.tools["read_file"](path="../../../etc/passwd")
        self.assertIn("escapes workspace", out)


class TestGrepFile(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self._tmp.name)
        articles = self.workspace / "articles"
        articles.mkdir(parents=True)
        (articles / "hit.md").write_text("line1\nneedle here\nline3\n", encoding="utf-8")
        (articles / "long.md").write_text(
            "prefix " + ("x" * 1000) + " Shadow Chancellor of the Exchequer " + ("y" * 1000),
            encoding="utf-8",
        )
        (articles / "psa.md").write_text(
            "Groupe PSA background. Carlos Tavares became CEO in 2014. "
            "Streiff was replaced by Corus Group chief executive Philippe Varin.",
            encoding="utf-8",
        )
        (articles / "cat.md").write_text(
            "Cat Stevens career. He hired an agent who arranged an audition with "
            "Chris Blackwell of Island Records. He later established his own record "
            "label, Jamal Records.",
            encoding="utf-8",
        )
        (articles / "summer.md").write_text(
            'Donna Summer signed with Casablanca Records in 1975. "She Works Hard for the '
            'Money" was released in 1983 on Mercury Records. '
            + ("catalog filler " * 30)
            + "Later Warner Bros. sister "
            "company, Atlantic Records, to sign Summer in the US after a European hit.",
            encoding="utf-8",
        )
        (articles / "gloria.md").write_text(
            "Grahame was married four times and had four children. Her first marriage "
            "was to actor Stanley Clements in August 1945. They divorced in June 1948.",
            encoding="utf-8",
        )
        (articles / "marilyn.md").write_text(
            "Born in Los Angeles, Monroe spent most of her childhood in foster homes "
            "and an orphanage before marrying James Dougherty at the age of 16.",
            encoding="utf-8",
        )
        (articles / "ed.md").write_text(
            'Autumn Variations was released on 29 September 2023. It is Sheeran\'s first '
            "studio album to be released through Gingerbread Man Records.",
            encoding="utf-8",
        )
        (articles / "rihanna.md").write_text(
            "Signed to Def Jam Recordings, Rihanna debuted with Music of the Sun. "
            "That May, she officially parted ways with Def Jam Recordings and "
            "transitioned fully to Roc Nation, the label that had been managing her "
            "career since October 2010. In 2016, it was confirmed that Rihanna would "
            "release her music through her own label, Westbury Road Entertainment, "
            "founded in 2005 and named after her childhood home in Barbados.",
            encoding="utf-8",
        )
        (articles / "surya.md").write_text(
            "Yadav started playing club cricket in Mumbai before he was selected for the "
            "Mumbai cricket team. Yadav plays for Mumbai in the Indian domestic cricket. "
            "Yadav plays for Mumbai Indians and has previously played for Kolkata Knight "
            "Riders in the Indian Premier League.",
            encoding="utf-8",
        )
        (articles / "ucsd.md").write_text(
            "UCSD's teams compete at the university's RIMAC facility, Triton Ballpark, "
            "and LionTree Arena. Price Center, often referred to as PC, is the main "
            "student hub on campus. Student media publications include Triton "
            "Television and the KSDT radio station.",
            encoding="utf-8",
        )
        (articles / "sonic.md").write_text(
            '"Sonic the Hedgehog 3" was released by Paramount Pictures in the United '
            "States on December 20. Produced by Original Film, Marza Animation Planet, "
            "and Blur Studio in association with Sega Sammy Group.",
            encoding="utf-8",
        )
        (articles / "starz.md").write_text(
            "On December 15, 2003, Lionsgate acquired Artisan Entertainment for "
            "$220 million. "
            "On July 26, 2007, Lionsgate bought a partial stake in independent film "
            "distribution company Roadside Attractions. On September 10, 2007, "
            "Lionsgate bought Mandate Pictures for $56.3 million, $44.3 million "
            "in cash and $12 million in stock.",
            encoding="utf-8",
        )
        (articles / "lluis.md").write_text(
            "Lluís Costa is a Spanish professional basketball player. After a brief "
            "spell with Debreceni EAC, a club of the Hungarian Nemzeti Bajnokság I/A, "
            "Costa returned to his former club FC Barcelona B in November 2019.",
            encoding="utf-8",
        )
        (articles / "greg.md").write_text(
            "Ostertag's contract expired in 2004, making him a free agent in the league. "
            "After nine seasons in Utah, he joined the Sacramento Kings. "
            "In December 2011, Ostertag returned to professional basketball. He signed "
            "with the Texas Legends of the NBA Development League. After playing 10 games "
            "with the Legends, however, he ended his comeback due to knee injury.",
            encoding="utf-8",
        )
        (articles / "dell.md").write_text(
            "Curry spent 10 seasons in Charlotte. Curry played one season for the "
            "Milwaukee Bucks before playing his final three seasons in the NBA for the "
            "Toronto Raptors.",
            encoding="utf-8",
        )
        (articles / "lorenzo.md").write_text(
            "On February 10, 2019, Guangzhou Loong Lions of the Chinese Basketball "
            "Association was reported to have signed Brown. On August 3, 2019, "
            "KK Crvena zvezda announced that they had signed Brown.",
            encoding="utf-8",
        )
        (articles / "mike.md").write_text(
            "Mike Scott played for several NBA teams. On February 6, 2019, Scott was "
            "traded, along with Tobias Harris and Boban Marjanović, to the Philadelphia "
            "76ers in exchange for Wilson Chandler.",
            encoding="utf-8",
        )
        (articles / "singapore.md").write_text(
            "On 7 March 2019, Nazri Nasir was appointed as interim head coach. "
            "During the 2026 FIFA World Cup qualification match, Singapore under "
            "new head coach Tsutomu Ogura played at home to China.",
            encoding="utf-8",
        )
        (articles / "philippines_women.md").write_text(
            "Guided by Marlon Maro, who returned as head coach of the team, the "
            "Philippines qualified for the 2022 AFC Women's Asian Cup. Alen Stajcic "
            "was appointed as head coach in October 2021. The Philippines hosted "
            "the 2022 AFF Women's Championship and clinched their first-ever title.",
            encoding="utf-8",
        )
        (articles / "saul.md").write_text(
            "Saul married Ahinoam, daughter of Ahimaaz, with whom he sired at least "
            "five sons (Jonathan, Abinadab, Malchishua, Ishvi and Ish-bosheth) and "
            "two daughters.",
            encoding="utf-8",
        )
        (articles / "oliver.md").write_text(
            "Stone also has a daughter, Tara Chong Stone, with his wife, Chong son "
            "Chong (Korean: 순중 정, alternately Westernized as Sun-jung Jung), whom "
            "he has been married to since 1996.",
            encoding="utf-8",
        )
        (articles / "sharon.md").write_text(
            "Ariel Sharon was an Israeli general and politician who served as the "
            "prime minister of Israel from March 2001 until April 2006.",
            encoding="utf-8",
        )
        (articles / "leipzig.md").write_text(
            "Leipzig head of government. The mayor was originally chosen by the city "
            "council. Wolfgang Tiefensee served from 1998 until his resignation in 2005. "
            "He was succeeded by fellow SPD politician Burkhard Jung, who was elected in "
            "January 2006 and re-elected in 2013 and 2020.",
            encoding="utf-8",
        )
        (articles / "portland.md").write_text(
            "Portland head of government. Since January 1, 2025, the city of Portland "
            "is governed by a mayor-council government system. Fred L. Peterson in "
            "1952 is the city's last elected Republican mayor, and no Republican has "
            "served as mayor even on an interim basis since Connie McCready held the "
            "post from 1979 to 1980. Mayor Neil Goldschmidt took office in 1972 as a "
            "proponent of downtown housing.",
            encoding="utf-8",
        )
        self.sandbox = AgentSandbox(self.workspace)
        self.fs = FsStore(self.workspace)
        self.episode = AgentEpisodeState()
        _, self.tools = make_tool_functions(
            sandbox=self.sandbox,
            fs_store=self.fs,
            episode=self.episode,
            toolset="hybrid_deep",
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_grep_file_only_on_hybrid_deep(self) -> None:
        self.assertIn("grep_file", self.tools)
        _, shell_tools = make_tool_functions(
            sandbox=self.sandbox,
            fs_store=self.fs,
            episode=self.episode,
            toolset="shell",
        )
        self.assertNotIn("grep_file", shell_tools)

    def test_grep_finds_pattern(self) -> None:
        out = self.tools["grep_file"](path="articles/hit.md", pattern="needle")
        self.assertIn("needle", out)

    def test_grep_supports_regex_without_shell_pipe_rejection(self) -> None:
        out = self.tools["grep_file"](path="articles/hit.md", pattern="needle|missing")
        self.assertIn("needle", out)
        self.assertNotIn("piped commands", out)

    def test_grep_returns_match_centered_snippet_for_long_lines(self) -> None:
        out = self.tools["grep_file"](
            path="articles/long.md",
            pattern="Shadow Chancellor",
        )
        self.assertIn("Shadow Chancellor of the Exchequer", out)
        self.assertTrue(out.startswith("1:..."))
        self.assertLess(len(out), 500)

    def test_hybrid_deep_search_adds_relation_snippet(self) -> None:
        out = self.tools["search_hybrid"](query="chairperson of Groupe PSA", k=1)
        self.assertIn("path=articles/psa.md", out)
        self.assertIn("snippet:", out)
        self.assertIn("chief executive Philippe Varin", out)

    def test_search_snippet_count_can_be_overridden(self) -> None:
        _, tools = make_tool_functions(
            sandbox=self.sandbox,
            fs_store=self.fs,
            episode=self.episode,
            toolset="hybrid_deep",
            search_snippets_per_hit=1,
        )
        out = tools["search_hybrid"](query="chairperson of Groupe PSA", k=1)
        self.assertEqual(out.count("snippet:"), 1)
        self.assertIn("chief executive Philippe Varin", out)

    def test_hybrid_deep_search_adds_signed_record_label_snippet(self) -> None:
        out = self.tools["search_hybrid"](query="Cat Stevens record label", k=1)
        self.assertIn("path=articles/cat.md", out)
        self.assertIn("snippet:", out)
        self.assertIn("Blackwell of Island Records", out)

    def test_hybrid_deep_search_adds_to_sign_record_label_snippet(self) -> None:
        out = self.tools["search_hybrid"](query="Donna Summer record label", k=2)
        self.assertIn("path=articles/summer.md", out)
        self.assertIn("Atlantic Records, to sign Summer", out)

    def test_hybrid_deep_search_prioritizes_album_release_record_label_snippet(self) -> None:
        _, tools = make_tool_functions(
            sandbox=self.sandbox,
            fs_store=self.fs,
            episode=self.episode,
            toolset="hybrid_deep",
            search_snippets_per_hit=1,
        )
        out = tools["search_hybrid"](
            query='Donna Summer She Works Hard for the Money 1983 record label',
            k=2,
        )
        self.assertIn("path=articles/summer.md", out)
        self.assertIn("released in 1983 on Mercury Records", out)
        self.assertNotIn("Atlantic Records, to sign Summer", out)

    def test_hybrid_deep_search_adds_first_husband_marriage_snippet(self) -> None:
        out = self.tools["search_hybrid"](query="Gloria Grahame first husband", k=5)
        self.assertIn("path=articles/gloria.md", out)
        self.assertIn("first marriage was to actor Stanley Clements", out)

    def test_hybrid_deep_search_adds_before_marrying_first_husband_snippet(self) -> None:
        out = self.tools["search_hybrid"](query="Marilyn Monroe first husband", k=5)
        self.assertIn("path=articles/marilyn.md", out)
        self.assertIn("before marrying James Dougherty", out)

    def test_hybrid_deep_search_prefers_westernized_spouse_alias(self) -> None:
        out = self.tools["search_hybrid"](query="William Oliver Stone spouse", k=1)
        self.assertIn("path=articles/oliver.md", out)
        self.assertIn("alternately Westernized as Sun-jung Jung", out)
        self.assertNotIn("with his wife, Chong son Chong", out)

    def test_hybrid_deep_search_adds_released_through_record_label_snippet(self) -> None:
        out = self.tools["search_hybrid"](query="Ed Sheeran record label September 2023", k=5)
        self.assertIn("path=articles/ed.md", out)
        self.assertIn("released through Gingerbread Man Records", out)

    def test_hybrid_deep_search_prioritizes_own_record_label_snippet(self) -> None:
        out = self.tools["search_hybrid"](query="Rihanna record label", k=5)
        self.assertIn("path=articles/rihanna.md", out)
        self.assertIn("own label, Westbury Road Entertainment", out)

    def test_hybrid_deep_search_adds_ipl_current_team_snippet(self) -> None:
        out = self.tools["search_hybrid"](query="Suryakumar Yadav Indian Premier League team", k=1)
        self.assertIn("path=articles/surya.md", out)
        self.assertIn("plays for Mumbai Indians", out)

    def test_hybrid_deep_search_adds_domestic_cricket_team_snippet(self) -> None:
        out = self.tools["search_hybrid"](query="Suryakumar Yadav domestic cricket team", k=1)
        self.assertIn("path=articles/surya.md", out)
        self.assertIn("selected for the Mumbai cricket team", out)

    def test_hybrid_deep_search_adds_radio_station_snippet(self) -> None:
        out = self.tools["search_hybrid"](query="UCSD radio station owned by", k=1)
        self.assertIn("path=articles/ucsd.md", out)
        self.assertIn("KSDT radio station", out)

    def test_hybrid_deep_search_adds_ballpark_facility_snippet(self) -> None:
        out = self.tools["search_hybrid"](
            query="University of California San Diego collegiate baseball facility",
            k=1,
        )
        self.assertIn("path=articles/ucsd.md", out)
        self.assertIn("Triton Ballpark", out)

    def test_hybrid_deep_search_adds_campus_event_building_snippet(self) -> None:
        out = self.tools["search_hybrid"](
            query="University of California San Diego campus building cultural events",
            k=1,
        )
        self.assertIn("path=articles/ucsd.md", out)
        self.assertIn("Price Center", out)

    def test_hybrid_deep_search_adds_production_company_snippet(self) -> None:
        out = self.tools["search_hybrid"](query="Blur Studio production company", k=5)
        self.assertIn("path=articles/sonic.md", out)
        self.assertIn("Produced by Original Film", out)
        self.assertIn("Blur Studio", out)

    def test_hybrid_deep_search_adds_acquisition_snippet(self) -> None:
        _, tools = make_tool_functions(
            sandbox=self.sandbox,
            fs_store=self.fs,
            episode=self.episode,
            toolset="hybrid_deep",
            search_snippets_per_hit=1,
        )
        out = tools["search_hybrid"](query="Starz Entertainment acquisition September 2007", k=5)
        self.assertIn("path=articles/starz.md", out)
        self.assertIn("bought Mandate Pictures", out)
        self.assertNotIn("acquired Artisan Entertainment", out)

    def test_hybrid_deep_search_uses_episode_question_for_release_snippet(self) -> None:
        self.episode.question = 'Which studio was responsible for the release of "Sonic the Hedgehog 3"?'
        out = self.tools["search_hybrid"](query="Sonic the Hedgehog 3 studio", k=5)
        self.assertIn("path=articles/sonic.md", out)
        self.assertIn("released by Paramount Pictures", out)

    def test_hybrid_deep_search_adds_sports_team_brief_spell_snippet(self) -> None:
        out = self.tools["search_hybrid"](
            query="member of sports team FC Barcelona B Hungarian November 2019",
            k=5,
        )
        self.assertIn("path=articles/lluis.md", out)
        self.assertIn("brief spell with Debreceni EAC", out)

    def test_hybrid_deep_search_adds_sports_team_joined_snippet(self) -> None:
        out = self.tools["search_hybrid"](
            query="professional basketball team Greg Ostertag join after Utah Jazz",
            k=5,
        )
        self.assertIn("path=articles/greg.md", out)
        self.assertIn("joined the Sacramento Kings", out)

    def test_hybrid_deep_search_prioritizes_returned_professional_basketball_snippet(self) -> None:
        _, tools = make_tool_functions(
            sandbox=self.sandbox,
            fs_store=self.fs,
            episode=self.episode,
            toolset="hybrid_deep",
            search_snippets_per_hit=1,
        )
        out = tools["search_hybrid"](
            query="Greg Ostertag December 2011 professional basketball team comeback knee injury",
            k=5,
        )
        self.assertIn("path=articles/greg.md", out)
        self.assertIn("signed with the Texas Legends", out)

    def test_hybrid_deep_search_adds_final_seasons_team_snippet(self) -> None:
        out = self.tools["search_hybrid"](
            query="professional basketball team Dell Curry final three seasons",
            k=5,
        )
        self.assertIn("path=articles/dell.md", out)
        self.assertIn("final three seasons in the NBA for the Toronto Raptors", out)

    def test_hybrid_deep_search_adds_sports_team_signed_date_snippet(self) -> None:
        out = self.tools["search_hybrid"](
            query="basketball team Lorenzo Brown August 2019",
            k=5,
        )
        self.assertIn("path=articles/lorenzo.md", out)
        self.assertIn("KK Crvena zvezda announced", out)

    def test_hybrid_deep_search_adds_dated_traded_team_snippet(self) -> None:
        out = self.tools["search_hybrid"](
            query="professional basketball team Mike Scott traded February 6 2019",
            k=5,
        )
        self.assertIn("path=articles/mike.md", out)
        self.assertIn("to the Philadelphia 76ers", out)

    def test_hybrid_deep_search_prioritizes_year_specific_head_coach_snippet(self) -> None:
        out = self.tools["search_hybrid"](
            query="Singapore national football team head coach 2026 qualification",
            k=5,
        )
        self.assertIn("path=articles/singapore.md", out)
        self.assertIn("under new head coach Tsutomu Ogura", out)

    def test_hybrid_deep_search_prioritizes_appointed_head_coach_snippet(self) -> None:
        episode = AgentEpisodeState()
        _, tools = make_tool_functions(
            sandbox=self.sandbox,
            fs_store=self.fs,
            episode=episode,
            toolset="hybrid_deep",
            search_snippets_per_hit=1,
        )
        out = tools["search_hybrid"](
            query="Philippines women's national football team 2022 AFF Women's Championship head coach",
            k=5,
        )
        self.assertIn("path=articles/philippines_women.md", out)
        self.assertIn("Alen Stajcic was appointed as head coach", out)
        self.assertNotIn("Buda Bautista was appointed as head coach", out)

    def test_hybrid_deep_search_adds_brother_sons_list_snippet(self) -> None:
        out = self.tools["search_hybrid"](
            query="son of Saul brother of Ish-bosheth",
            k=5,
        )
        self.assertIn("path=articles/saul.md", out)
        self.assertIn("Ishvi and Ish-bosheth", out)

    def test_hybrid_deep_search_adds_prime_minister_position_snippet(self) -> None:
        out = self.tools["search_hybrid"](
            query="Ariel Sharon position 2026-01-01",
            k=5,
        )
        self.assertIn("path=articles/sharon.md", out)
        self.assertIn("prime minister of Israel", out)

    def test_hybrid_deep_search_adds_current_mayor_successor_snippet(self) -> None:
        out = self.tools["search_hybrid"](query="head of government of Leipzig", k=5)
        self.assertIn("path=articles/leipzig.md", out)
        self.assertIn("Burkhard Jung", out)

    def test_hybrid_deep_search_prioritizes_future_portland_held_post_snippet(self) -> None:
        out = self.tools["search_hybrid"](
            query="head of government Portland Oregon 2026-01-01",
            k=5,
        )
        self.assertIn("path=articles/portland.md", out)
        self.assertIn("Connie McCready held the post", out)

    def test_path_escape_rejected(self) -> None:
        out = self.tools["grep_file"](path="../../outside.md", pattern="x")
        self.assertIn("escapes workspace", out)

    def test_multiline_pattern_rejected(self) -> None:
        out = self.tools["grep_file"](path="articles/hit.md", pattern="a\nb")
        self.assertIn("single line", out)


class TestSubmitAnswerValidation(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self._tmp.name)
        self.sandbox = AgentSandbox(self.workspace)
        self.fs = FsStore(self.workspace)
        self.episode = AgentEpisodeState()
        _, self.tools = make_tool_functions(
            sandbox=self.sandbox,
            fs_store=self.fs,
            episode=self.episode,
            toolset="shell",
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_invalid_submit_count_tracks_validation_errors(self) -> None:
        out = self.tools["submit_answer"](answer="")
        self.assertIn("empty answer", out)
        self.assertFalse(self.episode.done)
        self.assertEqual(self.episode.invalid_submit_count, 1)

    def test_not_found_submit_counts_as_invalid(self) -> None:
        out = self.tools["submit_answer"](answer="Not found")
        self.assertIn("generic non-answer", out)
        self.assertFalse(self.episode.done)
        self.assertEqual(self.episode.invalid_submit_count, 1)

    def test_apology_submit_counts_as_invalid(self) -> None:
        out = self.tools["submit_answer"](answer="I apologize, but I cannot find it.")
        self.assertIn("looks like explanation", out)
        self.assertFalse(self.episode.done)
        self.assertEqual(self.episode.invalid_submit_count, 1)

    def test_valid_submit_does_not_increment_invalid_count(self) -> None:
        out = self.tools["submit_answer"](answer="Project Cadmus")
        self.assertIn("Answer recorded", out)
        self.assertTrue(self.episode.done)
        self.assertEqual(self.episode.invalid_submit_count, 0)


if __name__ == "__main__":
    unittest.main()
