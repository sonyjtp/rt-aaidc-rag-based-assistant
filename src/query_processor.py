"""Query processing and context management for RAG assistant."""
import file_utils
from app_constants import CHAT_HISTORY, NO_CHAT_HISTORY, QUERY_AUGMENTATION_PATH
from config import DISTANCE_THRESHOLD
from log_manager import logger


class QueryProcessor:
    """Handles query augmentation, context building, and validation.

    Steps:
    1. Load query augmentation configuration from YAML file.
    2. Augment user queries with recent conversation context for follow-up questions.
    3. Build context strings based on distance thresholds.
    4. Validate the relevance of retrieved context using an LLM.

    Features:
    - Uses keywords to identify follow-up vs. new topic questions.
    - Limits context augmentation for new topic questions.
    - Employs LLM for semantic validation of context relevance.

    """

    def __init__(self, memory_manager=None, llm=None):
        """Initialize the query processor."""
        self.memory_manager = memory_manager
        self.llm = llm
        self.follow_up_keywords = []
        self.new_topic_keywords = []
        self._load_augmentation_config()

    def _load_augmentation_config(self) -> None:
        """Load query augmentation configuration from YAML file."""
        try:
            config = file_utils.load_yaml(QUERY_AUGMENTATION_PATH)
            self.follow_up_keywords = config.get("follow_up_keywords", [])
            self.new_topic_keywords = config.get("new_topic_keywords", [])
        except FileNotFoundError:
            logger.warning("query-augmentation.yaml not found. Using default keywords.")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Error loading augmentation config: {e}. Using defaults.")

    def augment_query_with_context(self, query: str) -> str:
        """
        Augment the user's query with recent conversation context.
        This helps resolve pronouns and references in follow-up questions.
        However, for new topic questions or very different queries, we avoid augmentation.

        Args:
            query: Original user query

        Returns:
            Augmented query that includes recent context (if appropriate)
        """
        if not self.memory_manager:
            logger.warning("Memory manager not available for query augmentation. Using original query.")
            return query
        result_query = query
        try:
            memory_vars = self.memory_manager.get_memory_variables()
            chat_history = memory_vars.get(CHAT_HISTORY, "")

            if chat_history and chat_history != NO_CHAT_HISTORY:
                # Check if this looks like a follow-up question or new topic

                query_lower = query.lower()
                is_follow_up = any(keyword in query_lower for keyword in self.follow_up_keywords)
                is_new_topic = any(keyword in query_lower for keyword in self.new_topic_keywords)

                # If question starts with "what are/is" or similar, it's likely a new topic
                # unless it contains follow-up keywords
                if is_new_topic and not is_follow_up:
                    # New topic question - return without augmentation. result_query remains as-is
                    logger.info("New topic question detected. Using original query for clean search.")

                elif is_follow_up:
                    logger.info("Follow-up question detected. Augmenting query with recent context.")

                    # For follow-ups, include only the last user question to avoid biasing search
                    # Extract the last user message from chat history
                    lines = chat_history.strip().split("\n")
                    last_user_question = None

                    # Try multiple common formats for user messages
                    user_prefixes = ["Human: ", "User: ", "Q: ", "Question: ", "USER: ", "HUMAN: "]

                    for line in reversed(lines):
                        if not line.strip():  # Skip empty lines
                            continue
                        for prefix in user_prefixes:
                            if line.startswith(prefix):
                                last_user_question = line[len(prefix) :].strip()
                                logger.debug(
                                    f"Found last user question with prefix '{prefix}': {last_user_question[:50]}..."
                                )
                                break
                        if last_user_question:
                            break

                    if last_user_question:
                        # Augment query with last user question
                        result_query = f"Previous question: {last_user_question}\n\nCurrent question: {query}"
                        logger.debug("Query augmented with last user question for follow-up")
                    else:
                        # Fallback to full context if we can't extract
                        logger.warning(
                            "Could not extract last user question from chat history. Using full context as fallback."
                        )
                        result_query = f"{chat_history}\n\nCurrent question: {query}"
                        logger.debug("Query augmented with full chat history context (fallback)")
                else:
                    # Default: treat as new topic for clean search. result_query remains as-is
                    logger.debug("Query treated as new topic - using original query for clean search")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(f"Could not augment query with context: {e}")
        return result_query

    @staticmethod
    def build_context(flat_docs: list[str], flat_distances: list[float]) -> str:
        """Assemble a context string only if the top result meets the distance threshold."""
        if flat_docs and (not flat_distances or flat_distances[0] <= DISTANCE_THRESHOLD):
            return "\n".join(flat_docs)
        return ""

    def validate_context(self, query: str, context: str, search_query: str = None) -> bool:
        """Return True if the context is relevant; False otherwise (conservative on error)."""
        try:
            return self._is_context_relevant_to_query(query, context, search_query)
        except Exception as e:
            logger.warning(f"Context relevance validation failed: {e}")
            return False

    def _is_context_relevant_to_query(self, query: str, context: str, search_query: str = None) -> bool:
        """
        Validate that retrieved context actually addresses the user's query.
        Uses the LLM to perform semantic validation.

        Args:
            query: User's question
            context: Retrieved document context
            search_query: Augmented search query (optional, for follow-up validation)

        Returns:
            True if context answers the query, False otherwise
        """
        if not context or not context.strip():
            return False

        if not self.llm:
            logger.warning("LLM not available for context validation, assuming relevant")
            return True

        try:
            # For follow-up questions, use the search query context for better validation
            if search_query and search_query != query:
                # This is a follow-up question with augmented context
                validation_prompt = (
                    f"Given the conversation context, does the following retrieved context "
                    f"contain relevant information related to the user's follow-up question?\n\n"
                    f"Conversation: {search_query}\n\n"
                    f"Retrieved Context: {context[:1000]}\n\n"
                    f"Respond with YES or NO."
                )
            else:
                # Regular question validation - use more lenient criteria
                validation_prompt = (
                    f"Does the following context contain information that is relevant to "
                    f"answering or addressing this question? Be inclusive - if the context "
                    f"discusses related topics, answer YES.\n\n"
                    f"Question: {query}\n\n"
                    f"Context: {context[:1000]}\n\n"
                    f"Answer: YES or NO"
                )

            result = self.llm.invoke(validation_prompt)
            # Handle both string and AIMessage responses
            result_str = str(result) if not isinstance(result, str) else result
            # For AIMessage objects, extract content attribute
            if hasattr(result, "content"):
                result_str = result.content

            # More lenient parsing: look for YES anywhere in response
            result_upper = result_str.upper()
            is_relevant = "YES" in result_upper and "NO" not in result_upper.split("YES")[0]
            logger.debug(f"Validation response: {result_str.strip()}")
            logger.info(f"Context relevance validation: {is_relevant}")
            return is_relevant
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(f"Error validating context relevance: {e}")
            # On error, assume context is relevant to avoid false negatives
            return True
